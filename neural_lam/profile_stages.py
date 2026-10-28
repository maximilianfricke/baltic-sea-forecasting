"""Opt-in CUDA-event timings and NVTX ranges, including checkpoint replay."""

import csv
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from statistics import mean

import torch


def graph_stage(name):
    """Select disjoint message-passing modules, not their nested children."""
    parts = name.split(".")
    if parts[-1] == "g2m_gnn":
        return "g2m"
    if parts[-1] == "m2g_gnn":
        return "m2g"
    groups = {
        "mesh_up_gnns", "mesh_down_gnns", "intra_level_gnns",
        "intra_up_gnns", "intra_down_gnns",
    }
    if len(parts) >= 2 and parts[-2] in groups and parts[-1].isdigit():
        return "mesh"
    if parts[-1] == "processor":
        return "mesh"
    return None


class StageTimings:
    def __init__(self, args):
        self.args = args
        self.active = False
        self.phase = "forward"
        self.pending = []
        self.rows = {"phases": [], "stages": []}
        self.restore = []
        self.handles = []
        self.open_phase = None
        self.step_pair = None

    def _start(self, kind, phase, stage="", module=""):
        pair = {
            "kind": kind, "phase": phase, "stage": stage, "module": module,
            "start": torch.cuda.Event(enable_timing=True),
            "end": torch.cuda.Event(enable_timing=True),
        }
        pair["start"].record()
        label = phase if kind == "phases" else f"{phase}/{stage}/{module}"
        torch.cuda.nvtx.range_push(label)
        return pair

    def _end(self, pair):
        pair["end"].record()
        torch.cuda.nvtx.range_pop()
        self.pending.append(pair)

    @contextmanager
    def scope(self, kind, stage="", module=""):
        if not self.active:
            yield
            return
        phase = (
            "forward" if kind == "phases" else
            "recompute" if self.phase == "backward" else "forward"
        )
        pair = self._start(kind, phase, stage, module)
        try:
            yield
        finally:
            # Non-reentrant checkpointing can stop replay inside a module.
            # A finally block keeps event pairs and NVTX nesting balanced.
            self._end(pair)

    def _wrap(self, obj, method, kind, stage="", name=""):
        original = getattr(obj, method)
        had_override = method in obj.__dict__

        @wraps(original)
        def timed(*args, **kwargs):
            with self.scope(kind, stage, name):
                return original(*args, **kwargs)

        setattr(obj, method, timed)
        self.restore.append((obj, method, original, had_override))

    def install(self, model, optimizers):
        self._wrap(model, "training_step", "phases")
        for name in ("embedd_all", "embedd_current"):
            if hasattr(model, name):
                self._wrap(model, name, "stages", "embeddings", name)
        for name, module in model.named_modules():
            stage = graph_stage(name)
            if stage:
                self._wrap(module, "forward", "stages", stage, name)
        for optimizer in optimizers:
            self.handles.append(optimizer.register_step_post_hook(
                self._optimizer_finished
            ))

    def begin_step(self, step):
        self.active = step >= 0
        self.step = step + 1
        self.phase = "forward"
        self.pending = []
        if self.active:
            self.step_pair = self._start("phases", "step")

    def begin_phase(self, phase):
        self.phase = phase
        if self.active:
            self.open_phase = self._start("phases", phase)

    def end_phase(self):
        if self.open_phase:
            self._end(self.open_phase)
            self.open_phase = None

    def _optimizer_finished(self, optimizer, args, kwargs):
        # Optimizer.step wraps the forward/backward closure in Lightning.
        # Start after that closure and stop at the actual optimizer post-hook.
        self.end_phase()

    def end_step(self):
        if self.step_pair:
            self._end(self.step_pair)
            self.step_pair = None

    def collect(self):
        # BaselineProfile synchronizes once per batch before this call.
        for pair in self.pending:
            row = {
                "hidden_dim": self.args.hidden_dim,
                "processor_layers": self.args.processor_layers,
                "step": self.step, "phase": pair["phase"],
                "cuda_ms": pair["start"].elapsed_time(pair["end"]),
            }
            if pair["kind"] == "stages":
                row.update(stage=pair["stage"], module=pair["module"])
            self.rows[pair["kind"]].append(row)
        self.pending = []

    def close(self):
        self.end_phase()
        self.end_step()
        self.active = False
        for handle in self.handles:
            handle.remove()
        self.handles = []
        for obj, method, original, had_override in reversed(self.restore):
            if had_override:
                setattr(obj, method, original)
            else:
                delattr(obj, method)
        self.restore = []

    def save(self, trainer, log_metrics=True):
        path = Path(self.args.profile_output)
        metrics = {}
        for kind, rows in self.rows.items():
            if not rows:
                continue
            target = path.with_suffix(f".{kind}.csv")
            target.parent.mkdir(parents=True, exist_ok=True)
            header = not target.exists() or target.stat().st_size == 0
            with target.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                if header:
                    writer.writeheader()
                writer.writerows(rows)
            per_step = defaultdict(float)
            for row in rows:
                label = row["phase"]
                if kind == "stages":
                    label += "/" + row["stage"]
                per_step[(label, row["step"])] += row["cuda_ms"]
            labels = {label for label, _ in per_step}
            steps = {row["step"] for row in rows}
            for label in labels:
                metrics[f"timing/{label}_ms"] = mean(
                    per_step[(label, step)] for step in steps
                )
            print(f"Stage timing data: {target}", flush=True)
        if metrics and log_metrics and trainer.logger:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

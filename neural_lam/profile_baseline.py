"""Small profiling callback; enabled by train_model --profile_steps N."""

import csv
import json
import time
from pathlib import Path
from statistics import mean

import pytorch_lightning as pl
import torch


class BaselineProfile(pl.Callback):
    def __init__(self, args):
        self.args = args
        self.completed = 0
        self.times = []
        self.allocated = 0
        self.reserved = 0
        self.range_open = False
        self.detail = None

    def on_train_start(self, trainer, pl_module):
        if getattr(self.args, "profile_stages", False):
            from .profile_stages import StageTimings

            self.detail = StageTimings(self.args)
            self.detail.install(pl_module, trainer.optimizers)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Batches have already been loaded and transferred to the GPU.
        torch.cuda.synchronize()
        self._peaks()
        if self.completed == self.args.profile_warmup:
            torch.cuda.nvtx.range_push("profile")
            self.range_open = True
        torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()
        if self.detail:
            self.detail.begin_step(self.completed - self.args.profile_warmup)

    def on_before_backward(self, trainer, pl_module, loss):
        if self.detail:
            self.detail.begin_phase("backward")

    def on_after_backward(self, trainer, pl_module):
        if self.detail:
            self.detail.end_phase()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self.detail:
            self.detail.begin_phase("optimizer")

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if self.detail:
            self.detail.end_step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        self._peaks()
        if self.completed >= self.args.profile_warmup:
            self.times.append(elapsed)
        if self.detail:
            self.detail.collect()
        self.completed += 1
        if self.completed == (
            self.args.profile_warmup + self.args.profile_steps
        ):
            self._close_range()

    def _peaks(self):
        self.allocated = max(
            self.allocated, torch.cuda.max_memory_allocated()
        )
        self.reserved = max(self.reserved, torch.cuda.max_memory_reserved())

    def _close_range(self):
        if self.range_open:
            torch.cuda.nvtx.range_pop()
            self.range_open = False

    def _write(self, trainer, model, status):
        self._peaks()
        self._close_range()
        args = self.args
        gib = 1024**3
        row = {
            "status": status,
            "gpu": torch.cuda.get_device_name(),
            "gpu_GiB": torch.cuda.get_device_properties(0).total_memory / gib,
            "torch": torch.__version__,
            "config": args.config_path,
            "model": args.model,
            "graph": args.graph,
            "hidden_dim": args.hidden_dim,
            "hidden_dim_grid": args.hidden_dim_grid,
            "hidden_dim_mesh_nodes": args.hidden_dim_mesh_nodes,
            "hidden_dim_edge": args.hidden_dim_edge,
            "processor_layers": args.processor_layers,
            "encoder_processor_layers": args.encoder_processor_layers,
            "prior_processor_layers": args.prior_processor_layers,
            "batch_size": args.batch_size,
            "ar_steps": args.ar_steps_train,
            "precision": args.precision,
            "kl_beta": args.kl_beta,
            "crps_weight": args.crps_weight,
            "checkpointing": model.grad_checkpointing,
            "parameters": sum(p.numel() for p in model.parameters()),
            "measured_steps": len(self.times),
            "step_ms": mean(self.times) * 1000 if self.times else "",
            # Includes warmup/Adam initialization; excludes failed requests.
            "peak_allocated_GiB": self.allocated / gib,
            "peak_reserved_GiB": self.reserved / gib,
        }
        path = Path(args.profile_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row)
            if header:
                writer.writeheader()
            writer.writerow(row)
        print("PROFILE " + json.dumps(row), flush=True)
        if status == "ok" and trainer.logger:
            trainer.logger.log_metrics(
                {
                    f"profile/{key}": row[key]
                    for key in (
                        "step_ms", "peak_allocated_GiB",
                        "peak_reserved_GiB", "parameters",
                    )
                },
                step=trainer.global_step,
            )

    def on_train_end(self, trainer, pl_module):
        if self.detail:
            self.detail.close()
            self.detail.save(trainer)
        status = (
            "ok" if len(self.times) == self.args.profile_steps
            else "incomplete"
        )
        self._write(trainer, pl_module, status)

    def on_exception(self, trainer, pl_module, exception):
        if self.detail:
            self.detail.close()
            self.detail.save(trainer, log_metrics=False)
        status = (
            "OOM" if isinstance(exception, torch.cuda.OutOfMemoryError)
            else type(exception).__name__
        )
        self._write(trainer, pl_module, status)

"""Peak memory and time of one GraphEFM inference step (no gradients).

Runs ``predict_step`` (prior, sample, decoder) once on one sample of the
configured dataset, with ``torch.no_grad``. On CUDA the peak is
``torch.cuda.max_memory_allocated``; on CPU a dispatch-mode tracker sums the
bytes of live tensors created during the step and records the operation at
the peak. Weights and inputs are reported separately.

Usage:
    python -m benchmarks.inference_memory --device cpu --output out.json -- \
        --config_path sample_data/global_ocean_1.yaml --model graph_efm \
        --graph global_cluster_graph --hidden_dim 64 ...
Any number of --hidden values can be given; each builds a fresh model.
"""

# Standard library
import argparse
import json
import sys
import time
import types
import weakref

# Third-party
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class _Captured(BaseException):
    """Stops train_model.main once the model exists."""


def build_model_and_batch(train_args):
    """Build the model of ``train_model`` arguments and one sample."""
    if sys.platform == "win32":
        sys.modules.setdefault("resource", types.ModuleType("resource"))
    # First-party
    from neural_lam import train_model

    captured = {}
    real_dm, real_models = train_model.WeatherDataModule, train_model.MODELS

    def dm_wrapper(*args, **kwargs):
        captured["dm"] = real_dm(*args, **kwargs)
        return captured["dm"]

    def model_wrapper(name):
        def build(*args, **kwargs):
            captured["model"] = real_models[name](*args, **kwargs)
            raise _Captured()

        return build

    train_model.WeatherDataModule = dm_wrapper
    train_model.MODELS = {name: model_wrapper(name) for name in real_models}
    try:
        train_model.main(train_args + ["--batch_size", "1"])
    except _Captured:
        pass
    finally:
        train_model.WeatherDataModule = real_dm
        train_model.MODELS = real_models
    model, dm = captured["model"], captured["dm"]
    dm.setup("fit")
    batch = torch.utils.data.default_collate([dm.train_dataset[0]])
    init_states, _, forcing, boundary, atmosphere, _ = batch
    prev_prev = init_states[:, 0] if init_states.shape[1] >= 2 else None
    args = (init_states[:, -1], prev_prev, forcing[:, 0], boundary[:, 0],
            atmosphere[:, 0])
    return model, args


class LiveBytes(TorchDispatchMode):
    """Bytes of live CPU tensor storages created inside the mode."""

    def __init__(self, external):
        super().__init__()
        self.external = external  # storages that existed before
        self.count, self.size = {}, {}
        self.current = self.peak = 0
        self.peak_op, self.peak_shapes = None, None

    def _release(self, ptr):
        self.count[ptr] -= 1
        if self.count[ptr] == 0:
            self.current -= self.size.pop(ptr)
            del self.count[ptr]

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            if not isinstance(t, torch.Tensor) or t.device.type != "cpu":
                continue
            storage = t.untyped_storage()
            ptr = storage.data_ptr()
            if ptr == 0 or ptr in self.external:
                continue
            if ptr not in self.count:
                self.count[ptr] = 0
                self.size[ptr] = storage.nbytes()
                self.current += storage.nbytes()
            self.count[ptr] += 1
            weakref.finalize(t, self._release, ptr)
        if self.current > self.peak:
            self.peak = self.current
            self.peak_op = str(func)
            self.peak_shapes = [
                tuple(t.shape) for t in tree_flatten(out)[0]
                if isinstance(t, torch.Tensor)
            ]
        return out


def measure(train_args, device):
    model, args = build_model_and_batch(train_args)
    model.eval().to(device)
    args = tuple(a.to(device) if a is not None else None for a in args)
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    weights += sum(b.numel() * b.element_size() for b in model.buffers())
    inputs = sum(a.numel() * a.element_size() for a in args if a is not None)
    result = {"parameters": sum(p.numel() for p in model.parameters()),
              "weights_bytes": weights, "inputs_bytes": inputs}
    with torch.no_grad():
        if device.type == "cuda":
            model.predict_step(*args)  # warm-up (kernels, workspaces)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            start = time.perf_counter()
            model.predict_step(*args)
            torch.cuda.synchronize()
            result["step_ms"] = (time.perf_counter() - start) * 1000
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["activation_peak_bytes"] = (
                torch.cuda.max_memory_allocated() - base
            )
        else:
            tensors = list(model.parameters()) + list(model.buffers())
            external = {t.untyped_storage().data_ptr() for t in tensors}
            external |= {a.untyped_storage().data_ptr()
                         for a in args if a is not None}
            tracker = LiveBytes(external)
            start = time.perf_counter()
            with tracker:
                model.predict_step(*args)
            result["step_ms"] = (time.perf_counter() - start) * 1000
            result["activation_peak_bytes"] = tracker.peak
            result["peak_op"] = tracker.peak_op
            result["peak_op_output_shapes"] = tracker.peak_shapes
            result["peak_allocated_bytes"] = weights + inputs + tracker.peak
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args, train_args = parser.parse_known_args()
    train_args = [a for a in train_args if a != "--"]
    device = torch.device(args.device)
    results = []
    for hidden in args.hidden:
        run_args = train_args + [
            "--hidden_dim", str(hidden), "--hidden_dim_grid", str(hidden),
            "--hidden_dim_mesh_nodes", str(hidden),
            "--hidden_dim_edge", str(hidden),
        ]
        entry = {"hidden": hidden}
        try:
            entry.update(measure(run_args, device))
            entry["status"] = "ok"
        except torch.OutOfMemoryError as e:
            entry.update(status="oom", error=str(e).splitlines()[0])
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps(entry), flush=True)
        results.append(entry)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"device": str(device), "train_args": train_args,
                   "results": results}, f, indent=2)


if __name__ == "__main__":
    main()

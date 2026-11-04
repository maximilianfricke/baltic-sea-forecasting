"""Measure the cost model (n(l), m(s) and u(s)) used by ``costs.py``.

Runs one forecast step (the gradient-checkpointed segment) of the model
configured by the given train_model arguments, on CPU with batch size 1, and
records every tensor saved for backward. Each saved tensor is attributed to
the node layer or edge set whose size matches one of its dimensions, giving
the values stored per node and per edge in units of hidden_dim. The uses
u(s) are counted with forward hooks on the message-passing layers.

Items of equal size (e.g. up edges, down edges and bottom mesh nodes when
every bottom mesh node has exactly one up edge) cannot be told apart by
size; their stored values are split evenly between them and reported.

Example:
    python -m neural_lam.graph_viewer.measure_weights \
        --output neural_lam/graph_viewer/weights/realistic_baseline.json \
        --name "Realistic Graph-EFM baseline" -- \
        --config_path sample_data/global_ocean_1.yaml --model graph_efm \
        --graph global_cluster_graph --hidden_dim 64 --processor_layers 2 \
        --ar_steps_train 4
"""

# Standard library
import datetime
import subprocess
import sys
import types
from argparse import ArgumentParser
from collections import defaultdict

# Third-party
import numpy as np
import torch

# Local
from .costs import CostModel
from .graph import load_viewer_graph


class _Captured(BaseException):
    """Stops train_model.main once the model exists (not an Exception, so
    the logger around main does not report it)."""


def _build_model_and_batch(train_args):
    if sys.platform == "win32":
        # benchmarking.py imports the Unix-only `resource` module, which is
        # not needed here
        sys.modules.setdefault("resource", types.ModuleType("resource"))
    # Local
    from .. import train_model

    captured = {}
    real_dm, real_models = train_model.WeatherDataModule, train_model.MODELS

    def dm_wrapper(*args, **kwargs):
        captured["dm"] = real_dm(*args, **kwargs)
        return captured["dm"]

    def model_wrapper(model_name):
        def build(*args, **kwargs):
            captured["model"] = real_models[model_name](*args, **kwargs)
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
    model.train()
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    init_states, target_states, forcing, boundary, atmosphere, _ = batch
    prev_states = init_states if model.input_steps >= 2 else init_states[:, -1:]
    step_args = (
        prev_states,
        target_states[:, 0],
        forcing[:, 0],
        boundary[:, 0],
        atmosphere[:, 0],
    )
    return model, step_args


def _edge_set_matcher(graph):
    """Identify an edge set from a model edge_index (indices may be offset)."""
    signatures = {
        es.name: (
            es.n_edges,
            len(np.unique(es.edge_index[0])),
            len(np.unique(es.edge_index[1])),
        )
        for es in graph.edge_sets
    }

    def match(edge_index):
        ei = edge_index.cpu().numpy()
        sig = (ei.shape[1], len(np.unique(ei[0])), len(np.unique(ei[1])))
        names = [name for name, s in signatures.items() if s == sig]
        if len(names) != 1:
            raise ValueError(f"Cannot identify edge set {sig}: {names}")
        return names[0]

    return match


def measure(model, step_args, graph, hidden_dim):
    match = _edge_set_matcher(graph)
    uses = defaultdict(int)
    for module in model.modules():
        edge_index = getattr(module, "edge_index", None)
        if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2:
            name = match(edge_index)

            def count(_module, _inputs, name=name):
                uses[name] += 1

            module.register_forward_pre_hook(count)

    # Only newly allocated storages count: not parameters, buffers or inputs
    static = {p.untyped_storage().data_ptr() for p in model.parameters()}
    static |= {b.untyped_storage().data_ptr() for b in model.buffers()}
    static |= {t.untyped_storage().data_ptr() for t in step_args}
    saved = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        ptr = storage.data_ptr()
        if ptr not in static and ptr not in saved:
            saved[ptr] = (
                storage.nbytes() // tensor.element_size(),
                tuple(tensor.shape),
            )
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model.compute_step_loss(*step_args)

    by_size = defaultdict(list)
    for layer in graph.layers:
        by_size[len(layer)].append(("node", layer.name))
    for es in graph.edge_sets:
        by_size[es.n_edges].append(("edge", es.name))

    stored, unattributed = defaultdict(float), 0.0
    for n_values, shape in saved.values():
        sizes = [d for d in shape if d in by_size]
        if not sizes:
            unattributed += n_values
            continue
        items = by_size[sizes[0]]
        for item in items:
            stored[item] += n_values / len(items)

    total = sum(stored.values()) + unattributed
    print(
        f"Stored for backward in one step: {total / 1e6:.1f} M values, "
        f"{100 * unattributed / total:.2f}% unattributed"
    )
    notes = [
        f"{', '.join(name for _, name in items)} have equal size "
        f"({size}); their stored values are split evenly"
        for size, items in by_size.items()
        if len(items) > 1
    ]
    if unattributed / total >= 1e-4:
        notes.append(
            f"{100 * unattributed / total:.2f}% of stored values matched no "
            "node layer or edge set and are left out"
        )
    for note in notes:
        print(f"Note: {note}")

    node = {
        layer.name: round(
            stored[("node", layer.name)] / len(layer) / hidden_dim, 3
        )
        for layer in graph.layers
    }
    edge = {
        es.name: round(
            stored[("edge", es.name)] / max(es.n_edges, 1) / hidden_dim, 3
        )
        for es in graph.edge_sets
    }
    uses = {es.name: uses[es.name] for es in graph.edge_sets}
    return node, edge, uses, notes, int(round(total))


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(input_args=None):
    parser = ArgumentParser(
        description=__doc__.splitlines()[0],
        usage="%(prog)s --output OUT --name NAME -- <train_model arguments>",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--name", type=str, required=True, help="Short name shown in viewer"
    )
    args, train_args = parser.parse_known_args(input_args)
    train_args = [a for a in train_args if a != "--"]

    def train_arg(name, default=None):
        return (
            train_args[train_args.index(name) + 1]
            if name in train_args
            else default
        )

    hidden_dim = int(train_arg("--hidden_dim", 64))
    graph = load_viewer_graph(
        train_arg("--config_path"), train_arg("--graph", "multiscale")
    )
    model, step_args = _build_model_and_batch(train_args)
    node, edge, uses, notes, total = measure(
        model, step_args, graph, hidden_dim
    )

    cost_model = CostModel(
        args.name,
        node,
        edge,
        uses,
        source={
            "train_model_args": " ".join(train_args),
            "graph": graph.name,
            "git_commit": _git_commit(),
            "measured": datetime.date.today().isoformat(),
            "batch_size": 1,
            "device": "cpu",
        },
        notes=notes,
        total_per_step_values=total,
    )
    cost_model.save(args.output)
    print(f"Saved cost model to {args.output}")
    for name, value in node.items():
        print(f"  n({name}) = {value:.1f}")
    for name in edge:
        print(f"  m({name}) = {edge[name]:.1f}   u = {uses[name]}")


if __name__ == "__main__":
    main()

"""P2P correctness against original AR4 training, without CPU placement code."""

# Standard library
import ast
from datetime import timedelta
from pathlib import Path

# Third-party
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# First-party
from message_passing.model import DistributedGraphEFM
from message_passing.partition import relation_plan
from message_passing.transport import exchange_halo
from tests.mp_fixture import ToyGraphEFM, _batch


def check_original_equivalence(
    rank, size, checkpointing, device=torch.device("cpu")
):
    torch.manual_seed(42)
    reference = ToyGraphEFM(checkpointing)
    owners = {
        "grid": torch.arange(8) % size,
        "mesh_0": torch.arange(4) % size,
        "mesh_1": (torch.arange(4) + 1) % size,
    }
    candidate = DistributedGraphEFM(reference, owners, device)
    reference.to(device)
    batch = _batch(device)
    local = candidate.local_batch(batch)
    opt_ref = torch.optim.AdamW(
        reference.parameters(), lr=0.001, betas=(0.9, 0.95), weight_decay=0.1
    )
    opt_dist = torch.optim.AdamW(
        candidate.parameters(), lr=0.001, betas=(0.9, 0.95), weight_decay=0.1
    )
    errors = {"predictions": 0.0, "gradients": 0.0, "parameters": 0.0}
    for update in range(2):
        reference.train()
        candidate.train()
        opt_ref.zero_grad(set_to_none=True)
        opt_dist.zero_grad(set_to_none=True)
        reference.predictions = []
        seed = 1000 + update
        torch.manual_seed(seed)
        expected_loss = reference.training_step(batch)
        expected_predictions = torch.stack(reference.predictions[:4], 1)
        expected_loss.backward()
        output = candidate(local, seed, capture_predictions=True)
        expected_local = expected_predictions.index_select(
            -2, candidate.nodes["grid"]
        )
        torch.testing.assert_close(
            output["predictions"], expected_local, rtol=3e-4, atol=3e-5
        )
        errors["predictions"] = max(
            errors["predictions"],
            float((output["predictions"] - expected_local).abs().max()),
        )
        global_loss = output["loss"].detach().clone()
        dist.all_reduce(global_loss)
        torch.testing.assert_close(
            global_loss, expected_loss, rtol=1e-5, atol=1e-4
        )
        output["loss"].backward()
        candidate.synchronize_gradients()
        for name, actual, expected in candidate.parameter_pairs(reference):
            assert (actual.grad is None) == (expected.grad is None), name
            if actual.grad is not None:
                scale = max(1.0, float(expected.grad.abs().max()))
                torch.testing.assert_close(
                    actual.grad / scale,
                    expected.grad / scale,
                    rtol=5e-4,
                    atol=5e-5,
                    msg=name,
                )
                errors["gradients"] = max(
                    errors["gradients"],
                    float((actual.grad - expected.grad).abs().max()) / scale,
                )
        opt_ref.step()
        opt_dist.step()
        for name, actual, expected in candidate.parameter_pairs(reference):
            torch.testing.assert_close(
                actual, expected, rtol=2e-4, atol=2e-4, msg=name
            )
            errors["parameters"] = max(
                errors["parameters"],
                float((actual - expected).abs().max().detach()),
            )
    reference.eval()
    torch.manual_seed(2001)
    with torch.no_grad():
        expected = reference.unroll_prediction(
            batch[0], batch[2], batch[3], batch[4]
        )[0]
        actual = candidate.forecast(local, 2001)
    torch.testing.assert_close(
        actual,
        expected.index_select(-2, candidate.nodes["grid"]),
        rtol=3e-4,
        atol=3e-5,
    )
    changed = list(local)
    changed[1] = torch.full_like(changed[1], 100.0)
    torch.testing.assert_close(
        candidate.forecast(tuple(changed), 2001), actual, rtol=0, atol=0
    )
    assert len(list(candidate.parameters())) == len(
        list(reference.parameters())
    )
    return errors


def worker(rank, size, rendezvous, checkpointing):
    torch.set_num_threads(1)  # Local CPU test resources only.
    torch.use_deterministic_algorithms(True)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=size,
        init_method=rendezvous,
        timeout=timedelta(seconds=45),
    )
    try:
        source_owners = torch.arange(8) % size
        destination_owners = torch.tensor([0, 0, 1, 1, 2, 3]) % size
        edges = torch.tensor(
            [[0, 1, 1, 2, 4, 6, 7, 0, 3, 5], [0, 0, 1, 1, 2, 2, 3, 3, 5, 5]]
        )
        plan = relation_plan(
            edges, source_owners, destination_owners, rank, size
        )
        full = torch.arange(48, dtype=torch.float64).reshape(2, 8, 3)
        features = full[:, plan.source_ids].clone().requires_grad_()
        result = exchange_halo(features, plan)
        torch.testing.assert_close(
            result,
            full[:, torch.cat((plan.source_ids, plan.halo_ids))],
            rtol=0,
            atol=0,
        )
        (
            result[:, len(plan.source_ids) :].square().sum() * (rank + 1)
            + features.square().sum()
        ).backward()
        weights = torch.ones(8, dtype=torch.float64)
        for peer in range(size):
            weights[
                relation_plan(
                    edges, source_owners, destination_owners, peer, size
                ).halo_ids
            ] += (peer + 1)
        torch.testing.assert_close(
            features.grad,
            2 * features.detach() * weights[plan.source_ids, None],
            rtol=0,
            atol=0,
        )
        check_original_equivalence(rank, size, checkpointing)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "size,checkpointing", [(1, True), (2, True), (4, True), (2, False)]
)
def test_original_ar4_p2p(tmp_path, size, checkpointing):
    mp.spawn(
        worker,
        args=(size, f"file://{tmp_path / 'init'}", checkpointing),
        nprocs=size,
        join=True,
    )


def test_scientific_sources_unchanged():
    # Standard library
    import subprocess

    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "76b30d7", "neural_lam"],
        text=True,
    ).splitlines()
    for name in tracked:
        expected = subprocess.check_output(
            ["git", "rev-parse", f"76b30d7:{name}"], text=True
        ).strip()
        actual = subprocess.check_output(
            ["git", "hash-object", name], text=True
        ).strip()
        assert actual == expected, name


def test_no_cpu_assignment_or_library_comparison():
    root = Path("message_passing")
    assert not (root / "launch.py").exists()
    assert not (root / "vendor").exists()
    for file in root.glob("*.py"):
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(
                node.func, ast.Attribute
            ):
                assert node.func.attr != "sched_setaffinity", file
    job = Path("jobs/message_passing.sbatch").read_text()
    assert (
        "--cpu-bind" not in job
        and "taskset" not in job
        and "numactl" not in job
    )


@pytest.mark.skipif(
    not Path("data/physical_repeat_16/global_ocean_1.yaml").exists(),
    reason="Optional local original data",
)
def test_real_h64_original_input_and_shared_sample_order():
    # Standard library
    from types import SimpleNamespace

    # First-party
    from message_passing.data import data_loader, local_batch, ownership
    from message_passing.workload import load_workload

    torch.set_num_threads(1)
    torch.manual_seed(42)
    args = SimpleNamespace(
        config_path=Path("data/physical_repeat_16/global_ocean_1.yaml"),
        graph="global_cluster_graph",
        hidden_dim=64,
        processor_layers=2,
        batch_size=2,
        ar_steps=4,
        lr=0.001,
        seed=42,
    )
    dataset, model, _ = load_workload(args)
    assert sum(p.numel() for p in model.parameters()) == 645020
    assert model.grad_checkpointing and model.interior_input_dim == 93
    assert [len(t) for t in model.mesh_static_features] == [8456, 2097]
    first = next(iter(data_loader(dataset, 2, 0, 42)))
    second = next(iter(data_loader(dataset, 2, 0, 42)))
    for actual, expected in zip(first, second, strict=True):
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=0, equal_nan=True
        )
    for size in (1, 2, 4):
        owners = ownership(model, size)
        ids = [torch.where(owners["grid"] == rank)[0] for rank in range(size)]
        assert sorted(torch.cat(ids).tolist()) == list(range(42348))
        for nodes in ids:
            selected = local_batch(first, nodes)
            torch.testing.assert_close(
                selected[0], first[0].index_select(-2, nodes), rtol=0, atol=0
            )

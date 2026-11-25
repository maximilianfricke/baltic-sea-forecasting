"""Original AR4 training/inference equivalence on 1/2/4 CPU ranks.

PhysicsNeMo's unequal list all-to-all requires NCCL. For CPU tests only that
lowest call is emulated with P2P; its packing/autograd code remains unchanged.
"""

# Standard library
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

# Third-party
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import set_checkpoint_early_stop

# First-party
from message_passing.model import DistributedGraphEFM
from message_passing.partition import relation_plan
from tests.mp_fixture import ToyGraphEFM, _batch


def emulated_all_to_all(outputs, inputs, group=None, async_op=False):
    assert not async_op
    operations = []
    for peer in range(dist.get_world_size()):
        if peer == dist.get_rank():
            outputs[peer].copy_(inputs[peer])
        else:
            if inputs[peer].numel():
                operations.append(dist.P2POp(dist.isend, inputs[peer], peer))
            if outputs[peer].numel():
                operations.append(dist.P2POp(dist.irecv, outputs[peer], peer))
    if operations:
        for request in dist.batch_isend_irecv(operations):
            request.wait()


def check_original_equivalence(
    rank, size, checkpointing, backend, device=torch.device("cpu")
):
    torch.manual_seed(42)
    reference = ToyGraphEFM(checkpointing)
    owners = {
        "grid": torch.arange(8) % size,
        "mesh_0": torch.arange(4) % size,
        "mesh_1": (torch.arange(4) + 1) % size,
    }
    candidate = DistributedGraphEFM(reference, owners, device, backend)
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
        with set_checkpoint_early_stop(False):
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
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=size,
        init_method=rendezvous,
        timeout=timedelta(seconds=120),
    )
    try:
        # First-party
        from message_passing.vendor.physicsnemo.distributed import (
            utils as pn_utils,
        )

        with patch.object(pn_utils.dist, "all_to_all", emulated_all_to_all):
            for backend in ("p2p", "physicsnemo"):
                # Unequal exchanges, repeated senders, and at K=4 a rank that
                # sends features but receives no halos for this relation.
                # First-party
                from message_passing.transport import HaloTransport

                source_owners = torch.arange(8) % size
                destination_owners = torch.tensor([0, 0, 1, 1, 2, 3]) % size
                edges = torch.tensor(
                    [
                        [0, 1, 1, 2, 4, 6, 7, 0, 3, 5],
                        [0, 0, 1, 1, 2, 2, 3, 3, 5, 5],
                    ]
                )
                plan = relation_plan(
                    edges, source_owners, destination_owners, rank, size
                )
                full = torch.arange(48, dtype=torch.float64).reshape(2, 8, 3)
                features = full[:, plan.source_ids].clone().requires_grad_()
                result = HaloTransport(backend)(features, plan)
                torch.testing.assert_close(
                    result,
                    full[:, torch.cat((plan.source_ids, plan.halo_ids))],
                    rtol=0,
                    atol=0,
                )
                (
                    result[:, len(plan.source_ids) :].square().sum()
                    * (rank + 1)
                    + features.square().sum()
                ).backward()
                weights = torch.ones(8, dtype=torch.float64)
                for peer in range(size):
                    peer_plan = relation_plan(
                        edges, source_owners, destination_owners, peer, size
                    )
                    weights[peer_plan.halo_ids] += peer + 1
                torch.testing.assert_close(
                    features.grad,
                    2 * features.detach() * weights[plan.source_ids, None],
                    rtol=0,
                    atol=0,
                )
                check_original_equivalence(rank, size, checkpointing, backend)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "size,checkpointing", [(1, True), (2, True), (4, True), (2, False)]
)
def test_ar4_original_equivalence(tmp_path, size, checkpointing):
    mp.spawn(
        worker,
        args=(size, f"file://{tmp_path / 'rendezvous'}", checkpointing),
        nprocs=size,
        join=True,
    )


def test_partition_preserves_edge_order_and_unique_halos():
    edges = torch.tensor([[0, 1, 1, 1, 2, 3], [0, 0, 0, 1, 2, 3]])
    owners = torch.tensor([0, 1, 0, 1])
    plan = relation_plan(edges, owners, owners, 0, 2)
    assert plan.edge_ids.tolist() == [0, 1, 2, 4]
    assert plan.halo_ids.tolist() == [1]
    assert plan.receive_counts == (0, 1)
    sources = torch.cat((plan.source_ids, plan.halo_ids))
    torch.testing.assert_close(
        sources[plan.edge_index[0]], edges[0, plan.edge_ids]
    )
    torch.testing.assert_close(
        plan.destination_ids[plan.edge_index[1]], edges[1, plan.edge_ids]
    )
    with pytest.raises(ValueError, match="empty rank"):
        relation_plan(edges, torch.zeros(4, dtype=torch.long), owners, 0, 2)


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


@pytest.mark.skipif(
    not Path("data/physical_repeat_16/global_ocean_1.yaml").exists(),
    reason="Optional physical data",
)
def test_real_graph_configuration_and_all_shards_preserve_original_inputs():
    # Standard library
    from types import SimpleNamespace

    # Third-party
    from torch.utils.data import default_collate

    # First-party
    from message_passing.data import local_cpu_batch, ownership, rank_dataset
    from message_passing.workload import load_workload

    torch.set_num_threads(1)
    torch.manual_seed(42)
    args = SimpleNamespace(
        config_path=Path("data/physical_repeat_16/global_ocean_1.yaml"),
        shard_root=Path("data/partitioned/physical_repeat_16"),
        graph="global_cluster_graph",
        ar_steps=4,
        hidden_dim=64,
        processor_layers=2,
        batch_size=2,
        lr=0.001,
        seed=42,
    )
    dataset, model, config = load_workload(args)
    assert sum(p.numel() for p in model.parameters()) == 645020
    assert model.grad_checkpointing and model.interior_input_dim == 93
    assert [len(t) for t in model.mesh_static_features] == [8456, 2097]
    full = default_collate([dataset[0], dataset[1]])
    for size in (1, 2, 4):
        owners = ownership(model, size)
        for rank in range(size):
            shard = rank_dataset(args, config, model, owners, rank, size)
            actual = default_collate([shard[0], shard[1]])
            expected = local_cpu_batch(
                full, torch.where(owners["grid"] == rank)[0], False
            )
            for left, right in zip(actual, expected, strict=True):
                if left.numel() or right.numel():
                    torch.testing.assert_close(
                        left, right, rtol=0, atol=0, equal_nan=True
                    )


def test_cpu_policies_use_only_allocated_ids():
    # First-party
    from message_passing.launch import cpu_shares

    assert cpu_shares([2, 5, 7, 9, 12, 13, 16, 19], 4, "disjoint") == [
        [2, 5],
        [7, 9],
        [12, 13],
        [16, 19],
    ]
    assert cpu_shares([2, 5, 7, 9], 2, "shared") == [[2, 5, 7, 9], [2, 5, 7, 9]]
    with pytest.raises(ValueError):
        cpu_shares([2], 2, "disjoint")
    with pytest.raises(ValueError):
        cpu_shares([2, 3], 1, "unknown")


def test_launcher_refuses_unverified_mask(monkeypatch):
    # First-party
    from message_passing import launch

    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setattr(
        launch.os, "sched_getaffinity", lambda pid: {2, 3}, raising=False
    )
    monkeypatch.setattr(
        "sys.argv", ["launch", "launch", "--ranks", "4", "--expected-cpus", "8"]
    )
    with pytest.raises(RuntimeError, match="does not match"):
        launch.main()


def test_launcher_exports_allocation_before_torchrun(monkeypatch):
    # Standard library
    import json
    import sys

    # First-party
    from message_passing import launch

    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setattr(
        launch.os, "sched_getaffinity", lambda pid: {2, 3, 8, 9}, raising=False
    )
    captured = []

    def execute(executable, arguments):
        captured.append((executable, arguments))
        raise RuntimeError("exec captured")

    monkeypatch.setattr(launch.os, "execv", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch",
            "launch",
            "--ranks",
            "2",
            "--expected-cpus",
            "4",
            "--cpu-policy",
            "shared",
            "--",
            "--config-path",
            "data.yaml",
        ],
    )
    with pytest.raises(RuntimeError, match="exec captured"):
        launch.main()
    assert json.loads(launch.os.environ["MP_JOB_CPUS"]) == [2, 3, 8, 9]
    assert launch.os.environ["OMP_NUM_THREADS"] == "1"
    assert "--nproc-per-node=2" in captured[0][1]
    assert captured[0][1][-2:] == ["--config-path", "data.yaml"]


def test_reports_reject_changed_controls_and_failed_validation(tmp_path):
    # Standard library
    import json

    # First-party
    from message_passing.report import summarize

    def result(i):
        return {
            "status": "completed",
            "performance_valid": True,
            "controls_sha256": "same",
            "controls": {"world_size": 2},
            "validation": {
                "p2p": [{"passed": True}],
                "physicsnemo": [{"passed": True}],
            },
            "ranks": [
                {"cpu_ready": {"main": [2, 3]}, "cpu_final": {"main": [2, 3]}}
            ],
            "cases": {
                b: {"summary": {"model_wall_ms": {"median": value + i}}}
                for b, value in (("p2p", 100), ("physicsnemo", 80))
            },
        }

    for i in range(4):
        (tmp_path / f"repeat-{i}.json").write_text(json.dumps(result(i)))
    report = summarize(tmp_path, 4)
    assert report["physicsnemo_speedup_over_p2p"] == pytest.approx(101.5 / 81.5)
    assert len(report["paired_speedups"]) == 4
    bad = result(3)
    bad["controls_sha256"] = "changed"
    (tmp_path / "repeat-3.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Controls differ"):
        summarize(tmp_path, 4)
    bad = result(3)
    bad["validation"]["p2p"][0]["passed"] = False
    (tmp_path / "repeat-3.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="equivalence failed"):
        summarize(tmp_path, 4)

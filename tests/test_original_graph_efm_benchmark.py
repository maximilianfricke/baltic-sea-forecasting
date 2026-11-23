"""Reference adapter must preserve the original loss, RNG and gradients."""

# Standard library
import copy
import json
from pathlib import Path
from types import SimpleNamespace

# Third-party
import pytest
import torch
import xarray as xr

from benchmarks.compare_model_parallel_real import _fingerprints_match
from benchmarks.model_parallel_real import (
    RankLocalBatchStream,
)
from benchmarks.original_graph_efm import (
    OriginalGraphEFMTraining,
    check_loss_trajectory,
    comparison_fingerprint,
    load_reference_workload,
    original_source_identity,
    validate_ocean_shard_order,
)
from tests.test_distributed_graph_efm_rollout import ToyGraphEFM, _batch


def test_reference_uses_frozen_original_sources():
    identity = original_source_identity()
    assert identity["commit"].startswith("76b30d7")
    assert len(identity["matching_git_blobs"]) == 13


def test_original_logging_adapter_preserves_ar4_loss_and_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    original = ToyGraphEFM(checkpointing=True)
    copied = copy.deepcopy(original)
    adapted = OriginalGraphEFMTraining(copied)
    batch = _batch(torch.device("cpu"))
    torch.manual_seed(43)
    expected = original.training_step(batch)
    expected.backward()
    torch.manual_seed(43)
    actual = adapted(batch, torch.default_generator)
    actual["loss"].backward()
    torch.testing.assert_close(actual["loss"], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        actual["loss"], actual["negative_likelihood"] + actual["kl"]
    )
    for (name, left), (_, right) in zip(
        original.named_parameters(), copied.named_parameters()
    ):
        torch.testing.assert_close(
            left.grad, right.grad, rtol=0, atol=0, msg=name
        )


@pytest.mark.skipif(
    not (
        Path("data/physical_repeat_16/global_ocean_1.yaml").exists()
        and Path("data/partitioned/physical_repeat_16/"
                 "geographic_rcb_ws1/manifest.json").exists()
    ),
    reason="local full-grid dataset and one-rank shard are optional",
)
def test_original_eight_worker_inputs_weights_and_ar4_loss_match():
    """Change the storage only; preserve batches, weights and original loss."""
    torch.set_num_threads(1)
    args = SimpleNamespace(
        config_path=Path("data/physical_repeat_16/global_ocean_1.yaml"),
        graph="global_cluster_graph",
        ar_steps=4,
        hidden_dim=64,
        processor_layers=2,
        batch_size=2,
        lr=0.001,
        seed=42,
        data_mode="rank-zero-broadcast",
        partitioned_data_root=Path("data/partitioned/physical_repeat_16"),
    )
    torch.manual_seed(42)
    full_dataset, full_model, full_metadata = load_reference_workload(args)
    args.data_mode = "rank-local"
    torch.manual_seed(42)
    shard_dataset, shard_model, shard_metadata = load_reference_workload(args)
    assert full_metadata["data_mode"] == "original-full-grid-zarr"
    assert shard_metadata["data_mode"] == "original-ocean-shard-zarr"
    assert shard_metadata["original_ocean_node_order_verified"]
    assert not shard_metadata["model_graph_partitioned"]
    assert type(full_model) is type(shard_model)
    assert sum(p.numel() for p in shard_model.parameters()) == 645020
    assert full_model.grad_checkpointing and shard_model.grad_checkpointing
    for name, expected in full_model.state_dict().items():
        torch.testing.assert_close(
            shard_model.state_dict()[name], expected,
            rtol=0, atol=0, equal_nan=True, msg=name,
        )

    batches = []
    for dataset in (full_dataset, shard_dataset):
        stream = RankLocalBatchStream(
            dataset, 2, 8, 2, torch.device("cpu"), seed=42
        )
        try:
            batches.append([stream.next(), stream.next()])
        finally:
            stream.iterator._shutdown_workers()
    for full, shard in zip(*batches):
        assert _fingerprints_match(
            comparison_fingerprint(full), comparison_fingerprint(shard)
        )
        for expected, actual in zip(full, shard):
            if not expected.numel() and not actual.numel():
                continue  # Only unused empty boundary extents differ.
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reference = json.loads(Path(
        "benchmark_results/model-parallel-real/baseline/"
        "original-ar4-workers8-prefetch2-4981/"
        "original-graph-efm-ar4-1gpu-4981.json"
    ).read_text())
    assert len(full_dataset) == len(shard_dataset) == 473
    assert _fingerprints_match(
        comparison_fingerprint(batches[1][0]),
        reference["preflight"]["batch_fingerprint"],
    )
    full_training = OriginalGraphEFMTraining(full_model.train())
    shard_training = OriginalGraphEFMTraining(shard_model.train())
    torch.manual_seed(100042)
    with torch.no_grad():
        expected = full_training(batches[0][0])
    torch.manual_seed(100042)
    with torch.no_grad():
        actual = shard_training(batches[1][0])
    for key in ("loss", "negative_likelihood", "kl"):
        assert torch.isfinite(actual[key])
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "coordinate", ["global_surface_index", "global_grid_index"]
)
def test_reference_rejects_reordered_shard_nodes(coordinate):
    coordinates = {"global_surface_index": [0, 1], "global_grid_index": [1, 3]}
    coordinates[coordinate] = coordinates[coordinate][::-1]
    store = xr.Dataset(coords={
        name: ("grid_index", values) for name, values in coordinates.items()
    })
    dataset = SimpleNamespace(
        datastore=SimpleNamespace(_ds=store), datastore_atmosphere=None,
    )
    model = SimpleNamespace(
        num_interior_nodes=2, surface_mask=[False, True, False, True],
    )
    with pytest.raises(ValueError, match=coordinate):
        validate_ocean_shard_order(dataset, model)


def test_reference_checks_measured_loss_and_respects_smoke_window():
    expected = [{"measurement_step": 0, "loss": 100.}]
    reference = {"records": expected, "workload": {"warmup_steps": 10}}
    assert check_loss_trajectory(expected, reference, 10)["exactly_equal"]
    assert not check_loss_trajectory(expected, reference, 0)["checked"]
    with pytest.raises(ValueError, match="trajectory mismatch"):
        check_loss_trajectory(
            [{"measurement_step": 0, "loss": 120.}], reference, 10
        )

"""Reference adapter must preserve the original loss, RNG and gradients."""

# Standard library
import copy
import json
from pathlib import Path
from types import SimpleNamespace

# Third-party
import pytest
import torch

from benchmarks.compare_model_parallel_real import _fingerprints_match
from benchmarks.model_parallel_real import (
    RankLocalBatchStream,
    _load_workload,
)
from benchmarks.original_graph_efm import (
    OriginalGraphEFMTraining,
    comparison_fingerprint,
    original_source_identity,
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
    not Path("data/physical_repeat_16/global_ocean_1.yaml").exists(),
    reason="local physical-repeat dataset is optional",
)
def test_original_full_grid_eight_worker_input_matches_archived_run():
    """Check the actual full-grid input path before scheduling its GPU run."""
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
    )
    dataset, _, _ = _load_workload(args)
    stream = RankLocalBatchStream(
        dataset, 2, 8, 2, torch.device("cpu"), seed=42
    )
    try:
        fingerprint = comparison_fingerprint(stream.next())
        reference_path = Path(
            "benchmark_results/model-parallel-real/full/"
            "ar4-workers8-prefetch2-4944-4949-4952/"
            "real-mp-max-ar4-ckpt-full-rank-local-1gpu-4944.json"
        )
        reference = json.loads(reference_path.read_text())
        assert len(dataset) == reference["workload"]["dataset_samples"] == 473
        assert _fingerprints_match(
            fingerprint, reference["preflight"]["batch_fingerprint"]
        )
    finally:
        stream.iterator._shutdown_workers()

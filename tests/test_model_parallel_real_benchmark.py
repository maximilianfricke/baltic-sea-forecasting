"""Tests for real-data model-parallel benchmark instrumentation."""

# Third-party
import pytest
import torch

# Local
from benchmarks import model_parallel_real


def test_data_loader_options_without_workers():
    options = model_parallel_real._data_loader_options(
        num_workers=0,
        prefetch_factor=4,
        device=torch.device("cpu"),
    )

    assert options == {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    }


def test_data_loader_options_with_spawn_prefetch():
    options = model_parallel_real._data_loader_options(
        num_workers=1,
        prefetch_factor=4,
        device=torch.device("cuda"),
    )

    assert options == {
        "num_workers": 1,
        "pin_memory": False,
        "persistent_workers": True,
        "multiprocessing_context": "spawn",
        "prefetch_factor": 4,
    }


def test_data_phase_synchronization_wait_is_measured(monkeypatch):
    barrier_calls = []
    timestamps = iter((10.0, 10.025))
    monkeypatch.setattr(
        model_parallel_real.time,
        "perf_counter",
        lambda: next(timestamps),
    )
    monkeypatch.setattr(
        model_parallel_real.dist,
        "barrier",
        lambda: barrier_calls.append(True),
    )

    wait_ms = model_parallel_real._synchronize_data_phase(
        torch.device("cpu"),
        world_size=2,
    )

    assert barrier_calls == [True]
    assert wait_ms == pytest.approx(25.0)


def test_single_rank_data_phase_skips_barrier(monkeypatch):
    timestamps = iter((10.0, 10.0))
    monkeypatch.setattr(
        model_parallel_real.time,
        "perf_counter",
        lambda: next(timestamps),
    )

    def fail_barrier():
        raise AssertionError("single-rank data phase must not call barrier")

    monkeypatch.setattr(model_parallel_real.dist, "barrier", fail_barrier)

    wait_ms = model_parallel_real._synchronize_data_phase(
        torch.device("cpu"),
        world_size=1,
    )

    assert wait_ms == 0.0


def test_default_workload_matches_archived_max_model(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "model_parallel_real.py", "--config-path", "unused.yaml",
        "--output", "unused.json",
    ])
    args = model_parallel_real._parse_args()
    model_args = model_parallel_real._model_args(args)
    assert args.num_workers == 8
    assert args.prefetch_factor == 2
    assert (args.warmup_steps, args.steps) == (10, 50)
    expected = {
        "hidden_dim": 64, "hidden_dim_grid": 64, "hidden_dim_edge": 64,
        "hidden_dim_mesh_nodes": 64, "hidden_layers": 1,
        "processor_layers": 2, "encoder_processor_layers": 1,
        "prior_processor_layers": 1, "input_steps": 2,
        "ar_steps_train": 4, "grad_checkpointing": True,
        "batch_size": 2, "loss": "wmse", "optimizer": "adamw",
        "lr": .001, "kl_beta": 1., "crps_weight": 0.,
        "prior_dist": "isotropic", "sample_obs_noise": 0,
        "output_std": False, "seed": 42,
    }
    for name, value in expected.items():
        assert getattr(model_args, name) == value, name


def test_rank_local_streams_share_shuffle_and_keep_partial_batch():
    dataset = torch.utils.data.TensorDataset(torch.arange(5))
    streams = [model_parallel_real.RankLocalBatchStream(
        dataset, 2, 0, 2, torch.device("cpu"), seed=42
    ) for _ in range(2)]
    orders = []
    for stream in streams:
        batches = [stream.next()[0] for _ in range(3)]
        assert [len(batch) for batch in batches] == [2, 2, 1]
        orders.append(torch.cat(batches))
    torch.testing.assert_close(orders[0], orders[1])
    assert sorted(orders[0].tolist()) == list(range(5))

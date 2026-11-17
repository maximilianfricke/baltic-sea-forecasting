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
        "pin_memory": True,
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

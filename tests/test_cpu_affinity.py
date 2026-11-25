"""Exercise Linux placement with simulated syscalls on macOS as well."""

import errno
import json
import os

import pytest

from benchmarks.cpu_affinity import AffinityGuard, MASK_ENV, POLICY, REPORT_ENV, launch, split_cpus


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_rank_masks_cover_only_the_noncontiguous_job_allocation(ranks):
    cpus = [6, 7, 10, 11, 16, 17, 22, 23][:2 * ranks]
    groups = split_cpus(cpus, ranks)
    assert all(len(group) == 2 for group in groups)
    assert sorted(cpu for group in groups for cpu in group) == cpus
    assert all(not set(a).intersection(b) for i, a in enumerate(groups) for b in groups[i + 1:])


def test_too_few_cpus_cannot_silently_oversubscribe():
    with pytest.raises(RuntimeError, match="at least one"):
        split_cpus([6, 7], 4)


@pytest.mark.parametrize("actual", [[6, 7], list(range(16))])
def test_launcher_rejects_narrow_and_unrestricted_masks(monkeypatch, tmp_path, actual):
    monkeypatch.setenv("SLURM_JOB_ID", "6121")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(actual), raising=False)
    with pytest.raises(RuntimeError, match="expected 8"):
        launch(8, tmp_path / "launch.json", ["-m", "torch.distributed.run"])
    assert not (tmp_path / "launch.json").exists()


def test_launcher_requires_a_slurm_step(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_STEP_ID", raising=False)
    with pytest.raises(RuntimeError, match="srun job step"):
        launch(8, tmp_path / "launch.json", [])


def test_launcher_records_verified_mask_before_loading_torch(monkeypatch, tmp_path):
    cpus = list(range(6, 14))
    monkeypatch.setenv("SLURM_JOB_ID", "6121")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("OMP_PROC_BIND", "true")
    monkeypatch.setenv("GOMP_CPU_AFFINITY", "6-7")
    monkeypatch.setenv("OMP_PLACES", "cores")
    # Register these keys with monkeypatch before launch changes them directly.
    for key in (MASK_ENV, REPORT_ENV, "KMP_AFFINITY"):
        monkeypatch.setenv(key, "before-launch-test")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(cpus), raising=False)
    executed = []
    monkeypatch.setattr(os, "execv", lambda executable, arguments: executed.append(arguments))
    output = tmp_path / "launch.json"
    launch(8, output, ["-m", "torch.distributed.run", "--nproc-per-node=4"])
    assert json.loads(output.read_text())["job_cpu_ids"] == cpus
    assert json.loads(os.environ[MASK_ENV]) == cpus
    assert os.environ["OMP_PROC_BIND"] == "false"
    assert os.environ["KMP_AFFINITY"] == "disabled"
    assert "GOMP_CPU_AFFINITY" not in os.environ and "OMP_PLACES" not in os.environ
    assert executed[0][1:] == ["-m", "torch.distributed.run", "--nproc-per-node=4"]


def fake_threads(monkeypatch):
    masks = {101: [6, 7], 102: [6, 7]}
    def get_mask(tid):
        return set(masks[101 if tid == 0 else tid])
    def set_mask(tid, cpus):
        masks[101 if tid == 0 else tid] = list(cpus)
    monkeypatch.setattr(os, "sched_getaffinity", get_mask, raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", set_mask, raising=False)
    monkeypatch.setattr(AffinityGuard, "thread_masks", staticmethod(lambda: {k:list(v) for k,v in masks.items()}))
    return masks


def test_restore_main_and_helper_threads_only_to_the_rank_share(monkeypatch, tmp_path):
    masks = fake_threads(monkeypatch)
    output = tmp_path / "rank.json"
    guard = AffinityGuard(list(range(6, 14)), rank=2, ranks=4, output=output)
    guard.checkpoint("after_nccl_init", repair=True)
    assert masks == {101: [10, 11], 102: [10, 11]}
    report = json.loads(output.read_text())
    assert report["rank_cpu_ids"] == [10, 11]
    assert report["checkpoints"][0]["main_before"] == [6, 7]
    assert report["checkpoints"][0]["passed"]


def test_affinity_regression_during_measurement_rejects_timing(monkeypatch, tmp_path):
    masks = fake_threads(monkeypatch)
    output = tmp_path / "rank.json"
    guard = AffinityGuard(list(range(6, 14)), rank=2, ranks=4, output=output)
    guard.checkpoint("after_warmup", repair=True)
    masks[102] = [10]
    with pytest.raises(RuntimeError, match="Reject this timing"):
        guard.checkpoint("after_measurement")
    assert masks[102] == [10]
    assert not json.loads(output.read_text())["checkpoints"][-1]["passed"]


def test_environment_cannot_substitute_an_unverified_cpu_list(monkeypatch, tmp_path):
    path = tmp_path / "launch.json"
    path.write_text(json.dumps({"policy": POLICY, "job_cpu_ids": [6, 7]}))
    monkeypatch.setenv(MASK_ENV, "[0, 1]")
    monkeypatch.setenv(REPORT_ENV, str(path))
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    with pytest.raises(RuntimeError, match="differs from the Slurm launch report"):
        AffinityGuard.from_environment()


def test_rank_reads_its_disjoint_share_from_the_launcher(monkeypatch, tmp_path):
    cpus = [6, 7, 10, 11, 16, 17, 22, 23]
    path = tmp_path / 'cpu-launch-0.json'
    path.write_text(json.dumps({'policy': POLICY, 'job_cpu_ids': cpus}))
    monkeypatch.setenv(MASK_ENV, json.dumps(cpus))
    monkeypatch.setenv(REPORT_ENV, str(path))
    monkeypatch.setenv('LOCAL_RANK', '3')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '4')
    guard = AffinityGuard.from_environment()
    assert guard.target == [22, 23]
    assert guard.output.name == 'cpu-launch-0-rank-3.json'


def test_exiting_helper_thread_does_not_break_a_repair(monkeypatch):
    masks = fake_threads(monkeypatch)
    original_set = os.sched_setaffinity
    def exiting(tid, cpus):
        if tid == 102:
            del masks[tid]
            raise ProcessLookupError(errno.ESRCH, "thread exited")
        original_set(tid, cpus)
    monkeypatch.setattr(os, "sched_setaffinity", exiting)
    guard = AffinityGuard(list(range(6, 14)), rank=2, ranks=4)
    guard.checkpoint("after_warmup", repair=True)
    assert masks == {101: [10, 11]}


def test_unmanaged_local_import_does_not_change_cpu_placement():
    guard = AffinityGuard()
    guard.checkpoint("local_import", repair=True)
    assert guard.report()["policy"] == "unmanaged" and guard.history == []

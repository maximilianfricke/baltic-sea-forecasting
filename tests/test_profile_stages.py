"""Exercise the real Lightning/checkpoint lifecycle without a GPU allocation."""

import csv
import time
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, TensorDataset

from neural_lam.profile_baseline import BaselineProfile
from neural_lam.profile_stages import graph_stage


class FakeEvent:
    def __init__(self, **kwargs):
        pass

    def record(self):
        self.timestamp = time.perf_counter()

    def elapsed_time(self, other):
        return (other.timestamp - self.timestamp) * 1000


class SmallGraph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.g2m_gnn = torch.nn.Linear(2, 2)
        self.intra_level_gnns = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.SiLU())
        ])
        self.m2g_gnn = torch.nn.Linear(2, 2)

    def forward(self, x):
        x = self.g2m_gnn(x)
        x = self.intra_level_gnns[0](x)
        return self.m2g_gnn(x)


class SmallModel(pl.LightningModule):
    grad_checkpointing = True

    def __init__(self):
        super().__init__()
        self.encoder = SmallGraph()

    def embedd_all(self, x):
        return x * 2

    def compute(self, x):
        return self.encoder(self.embedd_all(x)).square().mean()

    def training_step(self, batch):
        return checkpoint(self.compute, batch[0], use_reentrant=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.01)


def test_stages_preserve_checkpointed_training(tmp_path, monkeypatch):
    stack = []
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", stack.append)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", stack.pop)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 200)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "mock GPU")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _: SimpleNamespace(total_memory=1024),
    )
    args = SimpleNamespace(
        profile_output=str(tmp_path / "results.csv"), profile_warmup=1,
        profile_steps=3, profile_stages=True, config_path="test", model="test",
        graph="test", hidden_dim=2, hidden_dim_grid=None,
        hidden_dim_mesh_nodes=None, hidden_dim_edge=None, processor_layers=1,
        encoder_processor_layers=1, prior_processor_layers=1, batch_size=1,
        ar_steps_train=1, precision="32-true", kl_beta=0.1, crps_weight=0,
    )
    torch.manual_seed(42)
    baseline = SmallModel()
    detailed = SmallModel()
    detailed.load_state_dict(baseline.state_dict())
    loader = DataLoader(TensorDataset(torch.ones(2, 2)), batch_size=1)
    callback = BaselineProfile(args)
    for model, callbacks in ((baseline, []), (detailed, [callback])):
        trainer = pl.Trainer(
            accelerator="cpu", devices=1, max_steps=4, max_epochs=-1,
            logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False,
            callbacks=callbacks,
        )
        trainer.fit(model, loader)
    for expected, actual in zip(baseline.parameters(), detailed.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert stack == []  # Includes checkpoint replay's early-stop exception.
    assert "training_step" not in detailed.__dict__
    assert "forward" not in detailed.encoder.g2m_gnn.__dict__
    with (tmp_path / "results.phases.csv").open() as handle:
        phases = list(csv.DictReader(handle))
    assert len(phases) == 12
    assert {r["step"] for r in phases} == {"1", "2", "3"}
    assert {r["phase"] for r in phases} == {
        "forward", "backward", "optimizer", "step",
    }
    assert all(float(r["cuda_ms"]) >= 0 for r in phases)
    with (tmp_path / "results.stages.csv").open() as handle:
        stages = list(csv.DictReader(handle))
    assert {r["phase"] for r in stages} == {"forward", "recompute"}
    assert {r["stage"] for r in stages} == {
        "embeddings", "g2m", "mesh", "m2g",
    }


def test_stage_selection_does_not_double_count_children():
    assert graph_stage("decoder.g2m_gnn") == "g2m"
    assert graph_stage("decoder.m2g_gnn") == "m2g"
    assert graph_stage("decoder.intra_up_gnns.0") == "mesh"
    assert graph_stage("decoder.mesh_down_gnns.0") == "mesh"
    assert graph_stage("decoder.intra_up_gnns.0.module_0") is None
    assert graph_stage("decoder.g2m_gnn.edge_mlp") is None

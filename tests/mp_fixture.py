"""Small original two-level GraphEFM used only for equivalence checks."""

# Standard library
from types import SimpleNamespace

# Third-party
import pytorch_lightning as pl
import torch
from torch import nn

# First-party
from message_passing.workload import make_model_args
from neural_lam import metrics, utils
from neural_lam.models.graph_efm import GraphEFM
from neural_lam.models.hi_graph_latent_decoder import HiGraphLatentDecoder
from neural_lam.models.hi_graph_latent_encoder import HiGraphLatentEncoder


class ToyGraphEFM(GraphEFM):
    """Original training implementation with a locally constructed graph."""

    def __init__(self, checkpointing=True):
        pl.LightningModule.__init__(self)
        self.args = make_model_args(
            SimpleNamespace(
                ar_steps=4,
                batch_size=2,
                graph="toy",
                hidden_dim=4,
                processor_layers=2,
                lr=0.001,
                seed=42,
            )
        )
        self.args.latent_dim = 3
        self.input_steps = 2
        self.grad_checkpointing = checkpointing
        self.boundary_forced = False
        self.atmosphere_forced = True
        self.use_atmosphere_g2m = False
        self.concat_atmosphere = True
        self.hierarchical_graph = True
        self.embedd_m2m = True
        self.output_std = False
        self.sample_obs_noise = False
        self.use_density = False
        self.kl_beta = 1.0
        self.crps_weight = 0.0
        self.num_state_vars = 2
        self.predictions = []
        self.register_buffer("grid_static_features", torch.randn(8, 1))
        self.register_buffer(
            "loss_mask",
            torch.tensor([[1.0, 0.0], [0.5, 1.0], [1.0, 1.0], [1.0, 0.5]] * 2),
        )
        self.register_buffer("per_var_std", torch.tensor([1.0, 1.5]))
        for name in ("lower_upper", "lower", "upper"):
            self.register_buffer(
                "clamp_" + name + "_idx", torch.empty(0, dtype=torch.long)
            )
        ring = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
        )
        self.m2m_edge_index = (ring, ring.clone())
        up = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        self.mesh_up_edge_index = (up,)
        self.mesh_down_edge_index = (up.flip(0),)
        self.g2m_edge_index = torch.tensor(
            [
                list(range(8)) + list(range(8)),
                [i % 4 for i in range(8)] + [(i + 1) % 4 for i in range(8)],
            ]
        )
        self.m2g_edge_index = self.g2m_edge_index.flip(0)
        self.mesh_static_features = [torch.randn(4, 1), torch.randn(4, 1)]
        self.m2m_features = [torch.randn(8, 1), torch.randn(8, 1)]
        self.mesh_up_features = [torch.randn(4, 1)]
        self.mesh_down_features = [torch.randn(4, 1)]
        self.register_buffer("g2m_features", torch.randn(16, 1))
        self.register_buffer("m2g_features", torch.randn(16, 1))
        # 2 states x 2 channels + forcing + static + atmosphere = 7.
        self.interior_embedder = utils.make_mlp([7, 4, 4])
        self.interior_current_embedder = utils.make_mlp([9, 4, 4])
        self.g2m_embedder = utils.make_mlp([1, 4, 4])
        self.m2g_embedder = utils.make_mlp([1, 4, 4])
        for name, count in (
            ("mesh", 2),
            ("m2m", 2),
            ("mesh_up", 1),
            ("mesh_down", 1),
        ):
            setattr(
                self,
                name + "_embedders",
                nn.ModuleList(
                    [utils.make_mlp([1, 4, 4]) for _ in range(count)]
                ),
            )
        common = (
            3,
            self.g2m_edge_index,
            self.m2m_edge_index,
            self.mesh_up_edge_index,
            4,
            4,
            4,
            4,
        )
        self.prior_model = HiGraphLatentEncoder(
            *common,
            intra_level_layers=1,
            output_dist="isotropic",
            num_grid_con_mesh_nodes=4,
        )
        self.encoder = HiGraphLatentEncoder(
            *common,
            intra_level_layers=1,
            output_dist="diagonal",
            num_grid_con_mesh_nodes=4,
        )
        self.decoder = HiGraphLatentDecoder(
            self.g2m_edge_index,
            self.m2m_edge_index,
            self.m2g_edge_index,
            self.mesh_up_edge_index,
            self.mesh_down_edge_index,
            4,
            4,
            4,
            4,
            3,
            2,
            intra_level_layers=2,
            output_std=False,
            num_grid_con_mesh_nodes=4,
            num_interior_nodes=8,
        )
        self.loss = metrics.wmse

    def log_dict(self, *args, **kwargs):
        pass

    def compute_step_loss(self, *args):
        output = super().compute_step_loss(*args)
        self.predictions.append(output[2].detach())
        return output

    def to(self, device):
        # Production uses BufferList; keep toy list tensors on the same device.
        super().to(device)
        for name in (
            "mesh_static_features",
            "m2m_features",
            "mesh_up_features",
            "mesh_down_features",
        ):
            setattr(
                self,
                name,
                [tensor.to(device) for tensor in getattr(self, name)],
            )
        return self


def _batch(device):
    gen = torch.Generator().manual_seed(781)
    shapes = (
        (2, 2, 8, 2),
        (2, 4, 8, 2),
        (2, 4, 8, 1),
        (2, 4, 0, 0),
        (2, 4, 8, 1),
        (2, 4),
    )
    return tuple(
        torch.randn(*shape, generator=gen).to(device) for shape in shapes
    )

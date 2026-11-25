"""Distribute original GraphEFM layers; retain its original training algorithm.

The original-layer adapter design follows Nils Rottstegge's implementation at
nils-pnemo@69939c9. This version keeps original edge order and accepts either
P2P or PhysicsNeMo through one explicit transport interface.
"""

# Standard library
import copy

# Third-party
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import set_checkpoint_early_stop

# First-party
from neural_lam import utils
from neural_lam.interaction_net import FlexibleNet

# Local
from .partition import build_plans
from .transport import HaloTransport


class LocalLayer(nn.Module):
    def __init__(self, original, plan, transport):
        super().__init__()
        self.net = original
        self.plan, self.transport = plan, transport
        self.net.num_rec = len(plan.destination_ids)
        edges = plan.edge_index.clone()
        if not isinstance(original, FlexibleNet):
            edges[0] += self.net.num_rec
        self.net.edge_index = edges

    def forward(self, send, receive, edge, emb=None):
        return self.net(self.transport(send, self.plan), receive, edge, emb=emb)


class GlobalNoiseNormal(torch.distributions.Normal):
    """Draw reference noise, then select globally identified nodes."""

    def rsample(self, sample_shape=torch.Size()):
        shape = (
            *sample_shape,
            *self.loc.shape[:-2],
            self.global_nodes,
            self.loc.shape[-1],
        )
        noise = self.loc.new_empty(shape).normal_()
        return self.loc + self.scale * noise.index_select(-2, self.owned_nodes)


def _noise_hook(owned_nodes, global_nodes):
    def hook(module, inputs, distribution):
        if len(owned_nodes) == global_nodes:
            return distribution
        # The original encoder already validated these parameters. Preserve
        # its validation policy without running the same GPU checks twice.
        result = GlobalNoiseNormal(
            distribution.loc, distribution.scale, validate_args=False
        )
        result._validate_args = distribution._validate_args
        result.owned_nodes, result.global_nodes = owned_nodes, global_nodes
        return result

    return hook


class DistributedGraphEFM(nn.Module):
    """Spatial model parallelism, not independent data-parallel examples.

    Construct from a CPU original. Parameters stay replicated; node states and
    receiver-owned edges are sharded. The original loss is additive over grid
    and latent nodes, so shared parameter gradients must be summed.
    """

    def __init__(
        self,
        original,
        owners,
        device,
        backend="p2p",
        config=None,
        datastore=None,
    ):
        super().__init__()
        if (
            original.boundary_forced
            or original.use_atmosphere_g2m
            or original.sample_obs_noise
            or original.crps_weight
        ):
            raise NotImplementedError(
                "Supported: global ocean, concatenated atmosphere, WMSE+KL, "
                "no observation noise/CRPS"
            )
        if next(original.parameters()).device.type != "cpu":
            raise ValueError("Construct the adapter from a CPU original model")
        if not hasattr(original.prior_model, "g2m_gnn"):
            raise NotImplementedError("A learned graph prior is required")
        self.plans = build_plans(
            original, owners, dist.get_rank(), dist.get_world_size(), device
        )
        self.transport = HaloTransport(backend)
        self.nodes = {
            name: torch.where(value == dist.get_rank())[0].to(device)
            for name, value in owners.items()
        }
        self.model = copy.deepcopy(original)
        model = self.model
        if config is not None and datastore is not None:
            model.prepare_clamping_params(
                config, datastore
            )  # Rebind clamping closures to the copy.
        elif any(
            getattr(model, f"clamp_{key}_idx").numel()
            for key in ("lower_upper", "lower", "upper")
        ):
            raise ValueError(
                "Clamped models require their original config and datastore"
            )
        model.log_dict = lambda *args, **kwargs: None
        grid = self.nodes["grid"].cpu()
        for name in ("grid_static_features", "loss_mask", "interior_mask_bool"):
            if hasattr(model, name):
                setattr(model, name, getattr(model, name).index_select(0, grid))
        if hasattr(model, "interior_mask"):
            model.interior_mask = model.interior_mask[grid.numpy()]
        model.num_grid_nodes = model.num_total_grid_nodes = (
            model.num_interior_nodes
        ) = len(grid)
        for name in ("g2m", "m2g"):
            plan = self.plans[name]
            setattr(
                model,
                name + "_features",
                getattr(model, name + "_features")[plan.edge_ids.cpu()],
            )
            setattr(model, name + "_edge_index", plan.edge_index.cpu())
        model.mesh_static_features = utils.BufferList(
            [
                t[self.nodes[f"mesh_{i}"].cpu()]
                for i, t in enumerate(model.mesh_static_features)
            ],
            persistent=False,
        )
        for prefix, feature_attr, index_attr in (
            ("m2m_", "m2m_features", "m2m_edge_index"),
            ("up_", "mesh_up_features", "mesh_up_edge_index"),
            ("down_", "mesh_down_features", "mesh_down_edge_index"),
        ):
            features = [
                t[self.plans[f"{prefix}{i}"].edge_ids.cpu()]
                for i, t in enumerate(getattr(model, feature_attr))
            ]
            setattr(
                model,
                feature_attr,
                utils.BufferList(features, persistent=False),
            )
            setattr(
                model,
                index_attr,
                utils.BufferList(
                    [
                        self.plans[f"{prefix}{i}"].edge_index.cpu()
                        for i in range(len(features))
                    ],
                    persistent=False,
                ),
            )

        def wrap(layer, name):
            return LocalLayer(layer, self.plans[name], self.transport)

        for submodel in (model.prior_model, model.encoder, model.decoder):
            submodel.g2m_gnn = wrap(submodel.g2m_gnn, "g2m")
            if hasattr(submodel, "m2g_gnn"):
                submodel.m2g_gnn = wrap(submodel.m2g_gnn, "m2g")
            for attr, prefix in (
                ("mesh_up_gnns", "up_"),
                ("mesh_down_gnns", "down_"),
            ):
                for i, layer in enumerate(getattr(submodel, attr, ())):
                    getattr(submodel, attr)[i] = wrap(layer, f"{prefix}{i}")
            for attr in (
                "intra_level_gnns",
                "intra_up_gnns",
                "intra_down_gnns",
            ):
                for i, sequence in enumerate(getattr(submodel, attr, ())):
                    for name, layer in list(sequence.named_children()):
                        setattr(sequence, name, wrap(layer, f"m2m_{i}"))
        top = f"mesh_{len(owners) - 2}"
        for encoder in (model.encoder, model.prior_model):
            encoder.register_forward_hook(
                _noise_hook(self.nodes[top], len(owners[top]))
            )
        self.to(device)

    def local_batch(self, batch):
        return tuple(
            (
                t.index_select(-2, self.nodes["grid"].to(t.device))
                if i in (0, 1, 2, 4)
                else t
            )
            for i, t in enumerate(batch)
        )

    def forward(self, local_batch, seed, capture_predictions=False):
        """Return local additive loss; the runner reduces scalar metrics."""
        predictions = []
        original = self.model.compute_step_loss
        if capture_predictions:

            def capture(*args):
                output = original(*args)
                if len(predictions) < local_batch[1].shape[1]:
                    predictions.append(output[2].detach())
                return output

            self.model.compute_step_loss = capture
        devices = (
            [local_batch[0].device.index] if local_batch[0].is_cuda else []
        )
        try:
            with (
                torch.random.fork_rng(devices=devices),
                set_checkpoint_early_stop(False),
            ):
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    torch.cuda.default_generators[devices[0]].manual_seed(seed)
                loss = self.model.training_step(local_batch)
        finally:
            self.model.compute_step_loss = original
        return {
            "loss": loss,
            "predictions": torch.stack(predictions, 1) if predictions else None,
        }

    @torch.no_grad()
    def forecast(self, local_batch, seed):
        self.eval()
        initial, _, forcing, boundary, atmosphere, _ = local_batch
        devices = [initial.device.index] if initial.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            return self.model.unroll_prediction(
                initial, forcing, boundary, atmosphere
            )[0]

    def parameter_pairs(self, reference):
        originals = dict(reference.named_parameters())
        for name, parameter in self.model.named_parameters():
            yield name, parameter, originals[name.replace(".net.", ".")]

    def synchronize_gradients(self):
        if dist.get_world_size() == 1:
            return
        parameters = list(self.model.parameters())
        present = torch.tensor(
            [p.grad is not None for p in parameters],
            device=parameters[0].device,
            dtype=torch.int32,
        )
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
        active = [p for p, used in zip(parameters, present.tolist()) if used]
        if not active:
            raise RuntimeError("No model gradients were produced")
        flat = torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(
                    -1
                )
                for p in active
            ]
        )
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for p in active:
            value = flat[offset : offset + p.numel()].view_as(p)
            p.grad = value.clone() if p.grad is None else p.grad.copy_(value)
            offset += p.numel()

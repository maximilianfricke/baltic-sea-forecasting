"""Rank-local autoregressive GraphEFM training with per-step checkpointing.

This mirrors GraphEFM.training_step: sample the posterior, clamp the decoded
state, sum likelihood and KL over time, and feed predictions into the next
step without detaching. Only halo features (and their gradients) cross ranks;
no complete predicted grid is gathered between autoregressive steps.
"""

# Third-party
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

# First-party
from neural_lam.distributed_message_passing import select_owned


def embed_local_graph(model, partition, batch_size):
    """Apply the original graph embedders only to receiver-owned edges."""

    def embedded(embedder, features, indices):
        return model.expand_to_batch(
            embedder(features.index_select(0, indices)), batch_size
        )

    graph = {
        "g2m": embedded(
            model.g2m_embedder, model.g2m_features, partition.g2m.edge_ids
        ),
        "m2g": embedded(
            model.m2g_embedder, model.m2g_features, partition.m2g.edge_ids
        ),
    }
    if model.hierarchical_graph:
        graph["mesh"] = [
            embedded(embedder, features, nodes)
            for embedder, features, nodes in zip(
                model.mesh_embedders,
                model.mesh_static_features,
                partition.mesh_nodes,
            )
        ]
        graph["m2m"] = [
            (
                embedded(embedder, features, relation.edge_ids)
                if model.embedd_m2m
                else features.index_select(0, relation.edge_ids)
            )
            for embedder, features, relation in zip(
                (
                    model.m2m_embedders
                    if model.embedd_m2m
                    else [None] * len(partition.m2m)
                ),
                model.m2m_features,
                partition.m2m,
            )
        ]
        for name, relations in (
            ("mesh_up", partition.mesh_up),
            ("mesh_down", partition.mesh_down),
        ):
            graph[name] = [
                embedded(embedder, features, relation.edge_ids)
                for embedder, features, relation in zip(
                    getattr(model, name + "_embedders"),
                    getattr(model, name + "_features"),
                    relations,
                )
            ]
    else:
        graph["mesh"] = embedded(
            model.mesh_embedder,
            model.mesh_static_features,
            partition.mesh_nodes[0],
        )
        graph["m2m"] = embedded(
            model.m2m_embedder,
            model.m2m_features,
            partition.m2m[0].edge_ids,
        )
        graph["mesh_up"] = []
        graph["mesh_down"] = []
    return graph


class DistributedGraphEFMTraining(nn.Module):
    """Original GraphEFM training semantics on a spatial graph partition.

    ``frontend`` supplies the original embedders, loss, masks and clamping;
    its prior/encoder/decoder are replaced by ``core`` before construction.
    The supported global-ocean setup has no separate boundary graph nodes.
    """

    def __init__(self, frontend, core, partition):
        super().__init__()
        if frontend.boundary_forced or frontend.use_atmosphere_g2m:
            raise NotImplementedError(
                "distributed rollout requires the global-ocean configuration"
            )
        if not torch.equal(
            partition.full_grid_nodes, partition.interior_grid_nodes
        ):
            raise ValueError("rollout requires identical grid ownership")
        if frontend.crps_weight != 0 or frontend.kl_beta <= 0:
            raise ValueError("baseline requires CRPS weight 0 and KL beta > 0")
        if frontend.sample_obs_noise:
            raise ValueError("baseline disables observation-noise sampling")
        self.frontend = frontend
        self.core = core
        self.partition = partition

    def _step(self, previous_states, target, forcing, atmosphere, noise):
        model = self.frontend
        nodes = self.partition.interior_grid_nodes
        previous = previous_states[:, -1]
        previous_parts = [previous]
        if previous_states.shape[1] > 1:
            previous_parts.append(previous_states[:, 0])
        static = model.expand_to_batch(
            model.grid_static_features.index_select(0, nodes),
            previous.shape[0],
        )
        previous_parts.extend((forcing, static))
        current_parts = previous_parts + [target]
        if model.concat_atmosphere:
            previous_parts.append(atmosphere)
            current_parts.append(atmosphere)
        previous_emb = model.interior_embedder(
            torch.cat(previous_parts, dim=-1)
        )
        current_emb = model.interior_current_embedder(
            torch.cat(current_parts, dim=-1)
        )
        graph = embed_local_graph(model, self.partition, previous.shape[0])
        output = self.core.training_forward(
            previous_emb, current_emb, previous_emb, previous, graph, noise
        )
        mean = model.get_clamped_new_state(
            output["pred_mean"] - previous, previous
        )
        std = output["pred_std"] if model.output_std else model.per_var_std
        entry_losses = model.loss(
            mean,
            target,
            std,
            mask=select_owned(model.loss_mask, nodes),
            average_grid=False,
            sum_vars=False,
        )
        return entry_losses.sum(dim=(1, 2)), output["local_kl"], mean, std

    def forward(
        self,
        batch,
        generator=None,
        *,
        batch_is_local=True,
        latent_noise=None,
        return_predictions=False,
    ):
        """Return rank-local ELBO contributions for the complete rollout.

        ``latent_noise`` optionally supplies global B,T,N_latent,D noise for
        equivalence tests. Otherwise each rank draws the same global noise
        and selects its owned top-mesh nodes. Noise is drawn outside the
        checkpoint so recomputation never advances the explicit generator.
        """
        model = self.frontend
        initial, targets, forcing, _, atmosphere, _ = batch
        if not batch_is_local:
            nodes = self.partition.interior_grid_nodes
            initial, targets, forcing, atmosphere = (
                select_owned(tensor, nodes)
                for tensor in (initial, targets, forcing, atmosphere)
            )
        steps = targets.shape[1]
        if steps != model.args.ar_steps_train or forcing.shape[1] != steps:
            raise ValueError("batch rollout length differs from model setup")
        if initial.shape[-2] != self.partition.interior_grid_nodes.numel():
            raise ValueError("batch node count differs from graph partition")
        if initial.shape[1] != model.input_steps:
            raise ValueError("initial states differ from model input_steps")
        previous_states = initial
        nll_terms, kl_terms, predictions = [], [], []
        top_level = len(self.partition.mesh_nodes) - 1
        num_latent_nodes = (
            model.mesh_static_features[top_level].shape[0]
            if model.hierarchical_graph
            else model.mesh_static_features.shape[0]
        )
        latent_dim = model.args.latent_dim or model.args.hidden_dim
        for step in range(steps):
            if latent_noise is None:
                global_noise = torch.randn(
                    initial.shape[0],
                    num_latent_nodes,
                    latent_dim,
                    generator=generator,
                    device=initial.device,
                    dtype=initial.dtype,
                )
            else:
                global_noise = latent_noise[:, step]
            noise = select_owned(
                global_noise, self.partition.mesh_nodes[top_level]
            )
            arguments = (
                previous_states,
                targets[:, step],
                forcing[:, step],
                atmosphere[:, step],
                noise,
            )
            if (
                model.grad_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                # All ranks must replay the COMPLETE communication schedule.
                # Tensor-saving early-stop can otherwise stop at different
                # halo exchanges on different graph partitions and deadlock.
                with set_checkpoint_early_stop(False):
                    nll, kl, mean, std = checkpoint(
                        self._step, *arguments, use_reentrant=False
                    )
            else:
                nll, kl, mean, std = self._step(*arguments)
            nll_terms.append(nll)
            kl_terms.append(kl)
            if return_predictions:
                predictions.append(mean)
            feedback = model.apply_density_threshold(
                model.sample_next_state(mean, std)
            )
            previous_states = (
                torch.stack((previous_states[:, -1], feedback), dim=1)
                if model.input_steps >= 2
                else feedback.unsqueeze(1)
            )
        nll_per_step = torch.stack(nll_terms, dim=1)
        kl_per_step = torch.stack(kl_terms, dim=1)
        negative_likelihood = nll_per_step.sum(dim=1).mean()
        kl = kl_per_step.sum(dim=1).mean()
        return {
            "loss": negative_likelihood + model.kl_beta * kl,
            "negative_likelihood": negative_likelihood,
            "kl": kl,
            "negative_likelihood_per_step": nll_per_step.mean(dim=0),
            "kl_per_step": kl_per_step.mean(dim=0),
            "predictions": (
                torch.stack(predictions, dim=1)
                if return_predictions
                else None
            ),
        }

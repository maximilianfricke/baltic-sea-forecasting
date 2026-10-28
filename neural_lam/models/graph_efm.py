# Standard library
from typing import Union

# Third-party
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb

# First-party
from neural_lam import metrics, utils, vis

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from .ar_prob_model import ARProbModel
from .constant_latent_encoder import ConstantLatentEncoder
from .graph_latent_decoder import GraphLatentDecoder
from .graph_latent_encoder import GraphLatentEncoder
from .hi_graph_latent_decoder import HiGraphLatentDecoder
from .hi_graph_latent_encoder import HiGraphLatentEncoder


class GraphEFM(ARProbModel):
    """
    Graph-based Ensemble Forecasting Model
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        datastore_boundary: Union[BaseDatastore, None],
        datastore_atmosphere: Union[BaseDatastore, None],
        **kwargs,
    ):
        super().__init__(
            args,
            config=config,
            datastore=datastore,
            datastore_boundary=datastore_boundary,
            datastore_atmosphere=datastore_atmosphere,
            **kwargs,
        )

        self.prepare_clamping_params(config, datastore)

        assert (
            args.n_example_pred <= args.batch_size
        ), "Can not plot more examples than batch size in GraphEFM"
        self.sample_obs_noise = bool(args.sample_obs_noise)
        self.ensemble_size = args.ensemble_size
        self.kl_beta = args.kl_beta
        self.crps_weight = args.crps_weight
        self.num_latents_plot = args.num_latents_plot

        # Load graph with static features
        graph_dir_path = datastore.root_path / "graphs" / args.graph
        self.hierarchical_graph, graph_ldict = utils.load_graph(
            graph_dir_path=graph_dir_path,
            datastore=datastore,
        )
        for name, attr_value in graph_ldict.items():
            # NOTE: It would be good to rescale mesh node position features in
            # exactly the same way as grid node position static features.
            if name == "mesh_static_features":
                max_coord = datastore.get_xy("state", stacked=True).max()
                # Rescale by dividing by maximum coordinate in interior
                attr_value /= max_coord

            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Determine grid hidden dim
        if args.hidden_dim_grid is None:
            hidden_dim_grid = args.hidden_dim
        else:
            hidden_dim_grid = args.hidden_dim_grid

        # Determine edge hidden dim
        if args.hidden_dim_edge is None:
            hidden_dim_edge = hidden_dim_grid
        else:
            hidden_dim_edge = args.hidden_dim_edge

        # Determine mesh node hidden dim
        if args.hidden_dim_mesh_nodes is None:
            hidden_dim_mesh_nodes = args.hidden_dim
        else:
            hidden_dim_mesh_nodes = args.hidden_dim_mesh_nodes

        print(
            f"Using hidden_dim_grid={hidden_dim_grid}, "
            f"hidden_dim_mesh_nodes={hidden_dim_mesh_nodes}, "
            f"hidden_dim_edge={hidden_dim_edge}, hidden_dim={args.hidden_dim}"
        )

        # interior_dim from data + static
        self.g2m_edges, g2m_dim = self.g2m_features.shape
        self.m2g_edges, m2g_dim = self.m2g_features.shape

        print(f"g2m_dim.shape: {g2m_dim}, m2g_dim.shape: {m2g_dim}")

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        # For grid hidden dim
        self.grid_mlp_blueprint_end = [hidden_dim_grid] * (
            args.hidden_layers + 1
        )
        # For edge hidden dim
        self.edge_mlp_blueprint_end = [hidden_dim_edge] * (
            args.hidden_layers + 1
        )

        print(
            "GraphEFM, "
            f"self.interior_dim={self.interior_input_dim}, "
            f"self.boundary_dim={getattr(self, 'boundary_dim', None)}, "
            f"self.atmosphere_dim={getattr(self, 'atmosphere_dim', None)}, "
        )

        # Feature embedders for interior
        self.interior_embedder = utils.make_mlp(
            [self.interior_input_dim] + self.grid_mlp_blueprint_end
        )

        # We encode the current state as well, so we need to add more channels
        self.interior_current_embedder = utils.make_mlp(
            [self.interior_input_dim + self.num_state_vars]
            + self.grid_mlp_blueprint_end
        )

        # Define embedder for boundary and atmosphere nodes
        if self.boundary_forced:
            self.boundary_embedder = utils.make_mlp(
                [self.boundary_dim] + self.grid_mlp_blueprint_end,
            )
        if self.atmosphere_forced and self.use_atmosphere_g2m:
            self.atmosphere_embedder = utils.make_mlp(
                [self.atmosphere_dim] + self.grid_mlp_blueprint_end,
            )

        # Embedders for mesh
        self.g2m_embedder = utils.make_mlp(
            [g2m_dim] + self.edge_mlp_blueprint_end
        )
        self.m2g_embedder = utils.make_mlp(
            [m2g_dim] + self.edge_mlp_blueprint_end
        )

        if self.hierarchical_graph:
            # Print some useful info
            print("Loaded hierarchical graph with structure:")
            level_mesh_sizes = [
                mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
            ]
            num_levels = len(self.mesh_static_features)
            for level_index, level_mesh_size in enumerate(level_mesh_sizes):
                same_level_edges = self.m2m_features[level_index].shape[0]
                print(
                    f"level {level_index} - {level_mesh_size} nodes, "
                    f"{same_level_edges} same-level edges"
                )

                if level_index < (num_levels - 1):
                    up_edges = self.mesh_up_features[level_index].shape[0]
                    down_edges = self.mesh_down_features[level_index].shape[0]
                    print(f"  {level_index}<->{level_index + 1}")
                    print(f" - {up_edges} up edges, {down_edges} down edges")

            # Embedders
            # Assume all levels have same static feature dimensionality
            mesh_dim = self.mesh_static_features[0].shape[1]
            m2m_dim = self.m2m_features[0].shape[1]
            mesh_up_dim = self.mesh_up_features[0].shape[1]
            mesh_down_dim = self.mesh_down_features[0].shape[1]

            # Separate mesh node embedders for each level
            mesh_embedder_blueprint_0 = [hidden_dim_mesh_nodes] * (
                args.hidden_layers + 1
            )
            self.mesh_embedders = nn.ModuleList(
                [utils.make_mlp([mesh_dim] + mesh_embedder_blueprint_0)]
                + [
                    utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels - 1)
                ]
            )
            self.mesh_up_embedders = nn.ModuleList(
                [
                    utils.make_mlp([mesh_up_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels - 1)
                ]
            )
            self.mesh_down_embedders = nn.ModuleList(
                [
                    utils.make_mlp([mesh_down_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels - 1)
                ]
            )

            # If not using any processor layers, no need to embed m2m
            self.embedd_m2m = (
                max(
                    args.prior_processor_layers,
                    args.encoder_processor_layers,
                    args.processor_layers,
                )
                > 0
            )
            if self.embedd_m2m:
                self.m2m_embedders = torch.nn.ModuleList(
                    [
                        utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)
                        for _ in range(num_levels)
                    ]
                )
        else:
            # Single-level mesh
            num_mesh = self.mesh_static_features.shape[0]
            print(
                "Loaded non-hierarchical graph with structure:\n"
                f"  mesh nodes: {num_mesh}\n"
                f"  g2m edges: {self.g2m_edges}, m2g edges: {self.m2g_edges}, "
                f"m2m edges: {self.m2m_features.shape[0]}"
            )
            mesh_dim = self.mesh_static_features.shape[1]
            m2m_dim = self.m2m_features.shape[1]
            mesh_embedder_blueprint = [hidden_dim_mesh_nodes] * (
                args.hidden_layers + 1
            )
            self.mesh_embedder = utils.make_mlp(
                [mesh_dim] + mesh_embedder_blueprint
            )
            # m2m edge dim must equal processor hidden_dim
            self.m2m_embedder = utils.make_mlp(
                [m2m_dim] + self.mlp_blueprint_end
            )

        latent_dim = (
            args.latent_dim if args.latent_dim is not None else args.hidden_dim
        )

        # Specify dimensions of data
        print(
            "Loaded graph with "
            f"{self.num_total_grid_nodes + self.num_mesh_nodes} "
            f"nodes ({self.num_total_grid_nodes} grid, "
            f"{self.num_mesh_nodes} mesh)"
        )

        (
            self.num_interior_nodes,
            interior_static_dim,
        ) = self.grid_static_features.shape
        self.num_total_grid_nodes = self.num_interior_nodes

        # Prior
        if args.learn_prior:
            if self.hierarchical_graph:
                self.prior_model = HiGraphLatentEncoder(
                    latent_dim,
                    self.g2m_edge_index,
                    self.m2m_edge_index,
                    self.mesh_up_edge_index,
                    args.hidden_dim,
                    hidden_dim_grid,
                    hidden_dim_mesh_nodes,
                    hidden_dim_edge,
                    args.prior_processor_layers,
                    hidden_layers=args.hidden_layers,
                    output_dist=args.prior_dist,
                    num_grid_con_mesh_nodes=self.num_grid_con_mesh_nodes,
                )
            else:
                self.prior_model = GraphLatentEncoder(
                    latent_dim,
                    self.g2m_edge_index,
                    self.m2m_edge_index,
                    args.hidden_dim,
                    args.prior_processor_layers,
                    hidden_layers=args.hidden_layers,
                    output_dist=args.prior_dist,
                    hidden_dim_grid=hidden_dim_grid,
                    hidden_dim_mesh_nodes=hidden_dim_mesh_nodes,
                    hidden_dim_edge=hidden_dim_edge,
                    num_grid_con_mesh_nodes=self.num_grid_con_mesh_nodes,
                )
        else:
            self.prior_model = ConstantLatentEncoder(
                latent_dim,
                self.num_latent_mesh_nodes,
                output_dist=args.prior_dist,
            )

        # Enc. + Dec.
        if self.hierarchical_graph:
            # Encoder
            self.encoder = HiGraphLatentEncoder(
                latent_dim,
                self.g2m_edge_index,
                self.m2m_edge_index,
                self.mesh_up_edge_index,
                args.hidden_dim,
                hidden_dim_grid,
                hidden_dim_mesh_nodes,
                hidden_dim_edge,
                args.encoder_processor_layers,
                hidden_layers=args.hidden_layers,
                output_dist="diagonal",
                num_grid_con_mesh_nodes=self.num_grid_con_mesh_nodes,
            )
            # Decoder
            self.decoder = HiGraphLatentDecoder(
                self.g2m_edge_index,
                self.m2m_edge_index,
                self.m2g_edge_index,
                self.mesh_up_edge_index,
                self.mesh_down_edge_index,
                args.hidden_dim,
                hidden_dim_grid,
                hidden_dim_mesh_nodes,
                hidden_dim_edge,
                latent_dim,
                self.num_state_vars,
                args.processor_layers,
                hidden_layers=args.hidden_layers,
                output_std=bool(args.output_std),
                num_grid_con_mesh_nodes=self.num_grid_con_mesh_nodes,
                num_interior_nodes=self.num_interior_nodes,
            )
        else:
            self.encoder = GraphLatentEncoder(
                latent_dim,
                self.g2m_edge_index,
                self.m2m_edge_index,
                args.hidden_dim,
                args.encoder_processor_layers,
                hidden_layers=args.hidden_layers,
                output_dist="diagonal",
                hidden_dim_grid=hidden_dim_grid,
                hidden_dim_mesh_nodes=hidden_dim_mesh_nodes,
                hidden_dim_edge=hidden_dim_edge,
                num_grid_con_mesh_nodes=self.num_grid_con_mesh_nodes,
            )
            self.decoder = GraphLatentDecoder(
                self.g2m_edge_index,
                self.m2m_edge_index,
                self.m2g_edge_index,
                args.hidden_dim,
                latent_dim,
                hidden_dim_grid,
                self.num_state_vars,
                args.processor_layers,
                hidden_layers=args.hidden_layers,
                output_std=bool(args.output_std),
                hidden_dim_mesh_nodes=hidden_dim_mesh_nodes,
                hidden_dim_edge=hidden_dim_edge,
                num_interior_nodes=self.num_interior_nodes,
            )

    @property
    def num_mesh_nodes(self):
        """
        Get the total number of mesh nodes in the used mesh graph
        """
        if self.hierarchical_graph:
            num_mesh_nodes = sum(
                node_feat.shape[0] for node_feat in self.mesh_static_features
            )
        else:
            num_mesh_nodes = self.mesh_static_features.shape[0]
        return num_mesh_nodes

    @property
    def num_grid_con_mesh_nodes(self):
        """
        Get the total number of mesh nodes that have a connection to
        the grid (e.g. bottom level in a hierarchy)
        """
        if self.hierarchical_graph:
            return self.mesh_static_features[0].shape[0]  # Bottom level
        else:
            return self.mesh_static_features.shape[0]

    @property
    def num_latent_mesh_nodes(self):
        """
        Number of mesh sites for the variational latent (encoder / prior / KL).

        Single lev graph: same as total mesh nodes, hierarchical: top level only
        """
        if self.hierarchical_graph:
            return self.mesh_static_features[-1].shape[0]
        return self.mesh_static_features.shape[0]

    def sample_next_state(self, pred_mean, pred_std):
        """
        Sample state at next time step given Gaussian distribution.
        If self.sample_obs_noise is False, only return mean.

        pred_mean: (B, num_grid_nodes, d_state)
        pred_std: (B, num_grid_nodes, d_state) or
            None (if not output_std)

        Return:
        next_state: (B, num_grid_nodes, d_state)
        """
        if not self.output_std:
            pred_std = self.per_var_std  # (d_f,)

        if self.sample_obs_noise:
            return torch.distributions.Normal(pred_mean, pred_std).rsample()
            # (B, num_grid_nodes, d_state)

        return pred_mean  # (B, num_grid_nodes, d_state)

    def _add_optional_grid_embeddings(
        self,
        interior_emb,
        boundary_forcing,
        atmosphere_forcing,
    ):
        """
        Build embedding of full grid, optionally adding embeddings of
        boundary and atmosphere grid nodes.

        interior_emb: (B, num_interior_nodes, d_h)
        boundary_forcing: (B, num_boundary_nodes, boundary_dim)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_dim)

        Returns:
        grid_emb: (B, num_grid_nodes, d_h)
        """
        batch_size = interior_emb.shape[0]
        # List of all grid node embeddings
        grid_emb_list = [interior_emb]

        if self.boundary_forced:
            boundary_features = torch.cat(
                (
                    boundary_forcing,
                    self.expand_to_batch(
                        self.boundary_static_features, batch_size
                    ),
                ),
                dim=-1,
            )  # (B, num_boundary_nodes, interior_input_dim)
            boundary_emb = self.boundary_embedder(
                boundary_features
            )  # (B, num_boundary_nodes, d_h)
            grid_emb_list.append(boundary_emb)

        if self.atmosphere_forced and self.use_atmosphere_g2m:
            atmosphere_features = torch.cat(
                (
                    atmosphere_forcing,
                    self.expand_to_batch(
                        self.atmosphere_static_features, batch_size
                    ),
                ),
                dim=-1,
            )  # (B, num_atmosphere_nodes, interior_input_dim)

            atmosphere_emb = self.atmosphere_embedder(
                atmosphere_features
            )  # (B, num_atmosphere_nodes, d_h)
            grid_emb_list.append(atmosphere_emb)

        if len(grid_emb_list) == 1:
            # Only interior
            grid_emb = grid_emb_list[0]
        else:
            # NOTE: We here assume the order of grid node index is 1) interior,
            # 2) boundary, 3) atmosphere. This has to be followed also when
            # constructing g2m. Concatenates all existing embeddings.
            grid_emb = torch.cat(grid_emb_list, dim=1)
            # (B, num_grid_nodes, d_h)

        return grid_emb

    def embedd_current(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        boundary_forcing,
        atmosphere_forcing,
        current_state,
    ):
        """
        embed grid representation including current (target) state. Used as
        input to the encoder, which is conditioned also on the target.

        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1} (None if
            input_steps==1)
        forcing: (B, num_grid_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_dim)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_dim)
        current_state: (B, num_grid_nodes, feature_dim), X_{t+1}

        Returns:
        current_emb: (B, num_grid_nodes, d_h)
        """
        batch_size = prev_state.shape[0]
        interior_input_list = [prev_state]
        if prev_prev_state is not None:
            interior_input_list.append(prev_prev_state)
        interior_input_list.extend(
            [
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
                current_state,
            ]
        )
        if self.concat_atmosphere:
            interior_input_list.append(atmosphere_forcing)
        interior_features = torch.cat(interior_input_list, dim=-1)

        # Embed all features
        interior_emb = self.interior_current_embedder(
            interior_features
        )  # (B, num_interior_nodes, d_h)

        return self._add_optional_grid_embeddings(
            interior_emb, boundary_forcing, atmosphere_forcing
        )

    def embedd_all(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        boundary_forcing,
        atmosphere_forcing,
    ):
        """
        embed all node and edge representations

        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1} (None if
            input_steps==1)
        forcing: (B, num_grid_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_dim)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_dim)

        Returns:
        grid_emb: (B, num_grid_nodes, d_h)
        interior_emb: (B, num_interior_nodes, d_h)
        graph_embedding: dict with entries of shape (B, *, d_h)
        """
        batch_size = prev_state.shape[0]
        interior_input_list = [prev_state]
        if prev_prev_state is not None:
            interior_input_list.append(prev_prev_state)
        interior_input_list.extend(
            [
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ]
        )
        if self.concat_atmosphere:
            interior_input_list.append(atmosphere_forcing)
        interior_features = torch.cat(interior_input_list, dim=-1)

        # Embed all features
        interior_emb = self.interior_embedder(
            interior_features
        )  # (B, num_interior_nodes, d_h)

        full_grid_emb = self._add_optional_grid_embeddings(
            interior_emb, boundary_forcing, atmosphere_forcing
        )

        # Graph embedding
        graph_emb = {
            "g2m": self.expand_to_batch(
                self.g2m_embedder(self.g2m_features), batch_size
            ),  # (B, M_g2m, d_h)
            "m2g": self.expand_to_batch(
                self.m2g_embedder(self.m2g_features), batch_size
            ),  # (B, M_m2g, d_h)
        }

        if self.hierarchical_graph:
            graph_emb["mesh"] = [
                self.expand_to_batch(emb(node_static_features), batch_size)
                for emb, node_static_features in zip(
                    self.mesh_embedders,
                    self.mesh_static_features,
                )
            ]  # each (B, num_mesh_nodes[l], d_h)

            if self.embedd_m2m:
                graph_emb["m2m"] = [
                    self.expand_to_batch(emb(edge_feat), batch_size)
                    for emb, edge_feat in zip(
                        self.m2m_embedders, self.m2m_features
                    )
                ]
            else:
                # Need a placeholder otherwise, just use raw features
                graph_emb["m2m"] = list(self.m2m_features)

            graph_emb["mesh_up"] = [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(
                    self.mesh_up_embedders, self.mesh_up_features
                )
            ]
            graph_emb["mesh_down"] = [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(
                    self.mesh_down_embedders, self.mesh_down_features
                )
            ]
        else:
            graph_emb["mesh"] = self.expand_to_batch(
                self.mesh_embedder(self.mesh_static_features), batch_size
            )  # (B, num_mesh_nodes, d_h)
            graph_emb["m2m"] = self.expand_to_batch(
                self.m2m_embedder(self.m2m_features), batch_size
            )  # (B, M_m2m, d_h)

        return full_grid_emb, interior_emb, graph_emb

    def compute_step_loss(
        self,
        prev_states,
        current_state,
        forcing_features,
        boundary_forcing,
        atmosphere_forcing,
    ):
        """
        Perform forward pass and compute loss for one time step

        prev_states: (B, input_steps, num_grid_nodes, d_features)
        current_state: (B, num_grid_nodes, d_features) X^t
        forcing_features: (B, num_grid_nodes, d_forcing)
        boundary_forcing: (B, num_boundary_nodes, d_boundary)
        atmosphere_forcing: (B, num_atmosphere_nodes, d_atmosphere)
        """
        prev_state = prev_states[:, -1]
        prev_prev_state = (
            prev_states[:, 0] if prev_states.shape[1] > 1 else None
        )
        # embed all features
        grid_prev_emb, grid_prev_interior_emb, graph_emb = self.embedd_all(
            prev_state,
            prev_prev_state,
            forcing_features,
            boundary_forcing,
            atmosphere_forcing,
        )
        # embed also including current grid state, for encoder
        grid_current_emb = self.embedd_current(
            prev_state,
            prev_prev_state,
            forcing_features,
            boundary_forcing,
            atmosphere_forcing,
            current_state,
        )  # (B, num_grid_nodes, d_h)

        # Compute variational approximation (encoder)
        var_dist = self.encoder(
            grid_current_emb, graph_emb=graph_emb
        )  # Gaussian, (B, num_mesh_nodes, d_latent)

        # Compute likelihood
        last_state = prev_state
        likelihood_term, pred_mean, pred_std = self.estimate_likelihood(
            var_dist,
            current_state,
            last_state,
            grid_prev_emb,
            grid_prev_interior_emb,
            graph_emb,
        )
        if self.kl_beta > 0:
            # Compute prior
            prior_dist = self.prior_model(
                grid_prev_emb, graph_emb=graph_emb
            )  # Gaussian, (B, num_mesh_nodes, d_latent)

            # Compute KL
            kl_term = torch.sum(
                torch.distributions.kl_divergence(var_dist, prior_dist),
                dim=(1, 2),
            )  # (B,)
        else:
            # If beta=0, do not need to even compute prior nor KL
            kl_term = None  # Set to None to crash if erroneously used

        return likelihood_term, kl_term, pred_mean, pred_std

    def estimate_likelihood(
        self,
        latent_dist,
        current_state,
        last_state,
        grid_prev_emb,
        interior_grid_prev_emb,
        graph_emb,
    ):
        """
        Estimate (masked) likelihood using given distribution over
        latent variables

        latent_dist: distribution, (B, num_mesh_nodes, d_latent)
        current_state: (B, num_grid_nodes, d_state)
        last_state: (B, num_grid_nodes, d_state)
        interior_grid_prev_emb: (B, num__interior_grid_nodes, d_state)
        g2m_emb: (B, M_g2m, d_h)
        m2m_emb: (B, M_m2m, d_h)
        m2g_emb: (B, M_m2g, d_h)

        Returns:
        likelihood_term: (B,)
        pred_mean: (B, num_grid_nodes, d_state)
        pred_std: (B, num_grid_nodes, d_state) or (d_state,)
        """
        # Sample from variational distribution
        latent_samples = latent_dist.rsample()  # (B, num_mesh_nodes, d_latent)

        # Compute reconstruction (decoder)
        pred_mean, model_pred_std = self.decoder(
            grid_prev_emb,
            interior_grid_prev_emb,
            latent_samples,
            last_state,
            graph_emb,
        )  # both (B, num_grid_nodes, d_state)

        # Apply output clamping
        # (decoder adds residual: pred_mean = prev + delta)
        state_delta = pred_mean - last_state
        pred_mean = self.get_clamped_new_state(state_delta, last_state)

        if self.output_std:
            pred_std = model_pred_std  # (B, num_grid_nodes, d_state)
        else:
            # Use constant set std.-devs.
            pred_std = self.per_var_std  # (d_f,)

        # Compute likelihood (negative loss, exactly likelihood for nll loss)
        # Note: There are some round-off errors here due to float32
        # and large values
        entry_likelihoods = -self.loss(
            pred_mean,
            current_state,
            pred_std,
            mask=self.loss_mask,
            average_grid=False,
            sum_vars=False,
        )  # (B, num_grid_nodes', d_state)
        likelihood_term = torch.sum(entry_likelihoods, dim=(1, 2))  # (B,)
        return likelihood_term, pred_mean, pred_std

    def training_step(self, batch):
        """
        Train on single batch

        batch, containing:
        init_states: (B, 2, num_grid_nodes, d_state)
        target_states: (B, pred_steps, num_grid_nodes, d_state)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing), where
            index 0 corresponds to index 1 of init_states
        """
        (
            init_states,
            target_states,
            forcing,
            boundary_forcing,
            atmosphere_forcing,
            _,
        ) = batch

        if self.input_steps == 1:
            prev_state = init_states[:, 0]
            prev_prev_state = None
        else:
            prev_prev_state = init_states[:, 0]
            prev_state = init_states[:, 1]
        pred_steps = forcing.shape[1]

        loss_like_list = []
        loss_kl_list = []

        for i in range(pred_steps):
            target_state = target_states[:, i]  # (B, num_grid_nodes, d_state)

            if self.input_steps >= 2:
                prev_states_stacked = torch.stack(
                    (prev_prev_state, prev_state), dim=1
                )  # (B, 2, num_grid_nodes, d_state)
            else:
                prev_states_stacked = prev_state.unsqueeze(1)
                # (B, 1, num_grid_nodes, d_state)

            (
                loss_like_term,
                loss_kl_term,
                pred_mean,
                pred_std,
            ) = self.unroll_ckpt_func(
                self.compute_step_loss,
                prev_states_stacked,
                target_state,
                forcing[:, i],
                boundary_forcing[:, i],
                atmosphere_forcing[:, i],
            )
            # (B,), (B,), (B, num_grid_nodes, d_state),
            # pred_std is (B, num_grid_nodes, d_state) or (d_state)

            loss_like_list.append(loss_like_term)
            loss_kl_list.append(loss_kl_term)

            # Get predicted next state (sample or mean)
            predicted_state = self.sample_next_state(pred_mean, pred_std)

            # Apply density thresholding to the state fed back as input
            feedback_state = self.apply_density_threshold(predicted_state)

            # Update conditioning states
            if self.input_steps >= 2:
                prev_prev_state = prev_state
            prev_state = feedback_state

        # Compute final ELBO and loss, sum over time, mean over batch
        per_sample_likelihood = torch.sum(
            torch.stack(loss_like_list, dim=1), dim=1
        )  # (B,)
        mean_likelihood = torch.mean(per_sample_likelihood)
        log_dict = {
            "elbo_likelihood": mean_likelihood,
        }

        if self.kl_beta > 0:
            # Only compute full KL + ELBO if beta > 0
            per_sample_kl = torch.sum(
                torch.stack(loss_kl_list, dim=1), dim=1
            )  # (B,)
            mean_kl = torch.mean(per_sample_kl)
            elbo = mean_likelihood - mean_kl
            loss = -mean_likelihood + self.kl_beta * mean_kl

            log_dict["elbo"] = elbo
            log_dict["elbo_kl"] = mean_kl
        else:
            # Pure auto-encoder training
            loss = -mean_likelihood

        # Optionally sample trajectories and compute CRPS loss
        if self.crps_weight > 0:
            # Sample trajectories using prior
            pred_traj_means, pred_traj_stds = self.sample_trajectories(
                init_states,
                forcing,
                boundary_forcing,
                atmosphere_forcing,
                target_states,
                2,
            )
            # (B, S=2, pred_steps, num_grid_nodes, d_f), always 2 samples

            # Compute almost-fair CRPS without reduction
            crps_estimate = metrics.afcrps_ens(
                pred_traj_means,
                target_states,
                pred_traj_stds,
                average_grid=False,
                sum_vars=False,
                alpha=self.args.crps_alpha,
            )  # (B, pred_steps, num_grid_nodes, d_f)

            # Per-variable normalization and reduce
            crps_loss = torch.mean(
                metrics.mask_and_reduce_metric(
                    crps_estimate / self.per_var_std,
                    mask=self.loss_mask,
                    average_grid=True,
                    sum_vars=True,
                )
            )

            # Add onto loss
            loss = loss + self.crps_weight * crps_loss
            log_dict["crps_loss"] = crps_loss

        log_dict["train_loss"] = loss
        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss

    def predict_step(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        boundary_forcing,
        atmosphere_forcing,
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_interior_nodes, feature_dim), X_t
        prev_prev_state: (B, num_interior_nodes, feature_dim), X_{t-1} (None
            if input_steps==1)
        forcing: (B, num_interior_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_forcing_dim)
        """

        # embed all features
        grid_prev_emb, grid_prev_interior_emb, graph_emb = self.embedd_all(
            prev_state,
            prev_prev_state,
            forcing,
            boundary_forcing,
            atmosphere_forcing,
        )

        # Compute prior
        prior_dist = self.prior_model(
            grid_prev_emb, graph_emb=graph_emb
        )  # (B, num_mesh_nodes, d_latent)

        # Sample from prior
        latent_samples = prior_dist.rsample()
        # (B, num_mesh_nodes, d_latent)

        # Compute reconstruction (decoder)
        pred_mean, pred_std = self.decoder(
            grid_prev_emb,
            grid_prev_interior_emb,
            latent_samples,
            prev_state,
            graph_emb,
        )  # (B, num_grid_nodes, d_state)

        # TODO: Add option for pred_residual,
        # for now it is always True in the decoder

        # Apply output clamping
        # (decoder adds residual: pred_mean = prev + delta)
        state_delta = pred_mean - prev_state
        pred_mean = self.get_clamped_new_state(state_delta, prev_state)

        return self.sample_next_state(pred_mean, pred_std), pred_std

    def sample_trajectories(
        self,
        init_states,
        forcing_features,
        boundary_forcing,
        atmosphere_forcing,
        true_states,
        num_traj,
        use_encoder=False,
    ):
        """
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing)
        boundary_forcing: (B, pred_steps, num_boundary_nodes, d_boundary)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_forcing_dim)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        num_traj: S, number of trajectories to sample
        use_encoder: bool, if latent variables should be sampled from
            var. distribution

        Returns
        traj_means: (B, S, pred_steps, num_grid_nodes, d_f)
        traj_stds: (B, S, pred_steps, num_grid_nodes, d_f) or (d_f)
        """

        if use_encoder:
            traj_list = [
                self.unroll_prediction_vi(
                    init_states,
                    forcing_features,
                    boundary_forcing,
                    atmosphere_forcing,
                    true_states,
                )
                for _ in range(num_traj)
            ]
        else:
            traj_list = [
                self.unroll_prediction(
                    init_states,
                    forcing_features,
                    boundary_forcing,
                    atmosphere_forcing,
                )
                for _ in range(num_traj)
            ]
        # List of tuples, each containing
        # mean: (B, pred_steps, num_grid_nodes, d_f) and
        # std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        traj_means = torch.stack(
            [pred_pair[0] for pred_pair in traj_list], dim=1
        )
        if self.output_std:
            traj_stds = torch.stack(
                [pred_pair[1] for pred_pair in traj_list], dim=1
            )
        else:
            traj_stds = self.per_var_std

        return traj_means, traj_stds

    def unroll_prediction_vi(
        self,
        init_states,
        forcing,
        boundary_forcing,
        atmosphere_forcing,
        true_states,
    ):
        """
        Roll out prediction, sampling latent var. from variational
        encoder distribution

        init_states: (B, input_steps, num_grid_nodes, d_f)
        forcing: (B, pred_steps, num_grid_nodes, d_forcing)
        boundary_forcing: (B, pred_steps, num_boundary_nodes, d_boundary)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_forcing_dim)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        """
        if self.input_steps == 1:
            prev_state = init_states[:, 0]
            prev_prev_state = None
        else:
            prev_prev_state = init_states[:, 0]
            prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing.shape[1]

        for i in range(pred_steps):
            # Compute 1-step prediction, but using encoder
            step_forcing = forcing[:, i]
            step_boundary_forcing = boundary_forcing[:, i]
            step_atmosphere_forcing = atmosphere_forcing[:, i]
            current_state = true_states[:, i]

            # embed all features
            grid_prev_emb, grid_prev_interior_emb, graph_emb = self.embedd_all(
                prev_state,
                prev_prev_state,
                step_forcing,
                step_boundary_forcing,
                step_atmosphere_forcing,
            )

            # embed also including current grid state, for encoder
            grid_current_emb = self.embedd_current(
                prev_state,
                prev_prev_state,
                step_forcing,
                step_boundary_forcing,
                step_atmosphere_forcing,
                current_state,
            )

            # Compute variational distribution
            var_dist = self.encoder(
                grid_current_emb, graph_emb=graph_emb
            )  # Gaussian, (B, num_mesh_nodes, d_latent)

            # Sample from var. dist.
            latent_samples = var_dist.rsample()
            # (B, num_mesh_nodes, d_latent)

            # Compute reconstruction (decoder)
            pred_mean, pred_std = self.decoder(
                grid_prev_emb,
                grid_prev_interior_emb,
                latent_samples,
                prev_state,
                graph_emb,
            )  # (B, num_grid_nodes, d_state)

            # Apply output clamping
            state_delta = pred_mean - prev_state
            pred_mean = self.get_clamped_new_state(state_delta, prev_state)

            new_state = self.sample_next_state(pred_mean, pred_std)
            # pred_state: (B, num_grid_nodes, d_f)
            # pred_std: (B, num_grid_nodes, d_f) or None

            prediction_list.append(new_state)
            if self.output_std:
                pred_std_list.append(pred_std)

            # Apply density thresholding to the state fed back as input
            feedback_state = self.apply_density_threshold(new_state)

            # Update conditioning states
            if self.input_steps >= 2:
                prev_prev_state = prev_state
            prev_state = feedback_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std

    def ensemble_step(self, batch):
        """
        Perform ensemble forecast and compute basic metrics.
        Override base to pass true_states and num_traj to
        GraphEFM.sample_trajectories.
        """
        (
            init_states,
            target_states,
            forcing_features,
            boundary_forcing,
            atmosphere_forcing,
            _,
        ) = batch

        traj_means, _ = self.sample_trajectories(
            init_states,
            forcing_features,
            boundary_forcing,
            atmosphere_forcing,
            target_states,
            self.ensemble_size,
            use_encoder=False,
        )
        trajectories = traj_means  # (B, S, pred_steps, num_grid_nodes, d_f)

        spread_squared_batch = metrics.spread_squared(
            trajectories,
            target_states,
            None,
            mask=self.loss_mask,
            sum_vars=False,
        )
        ens_mean = torch.mean(trajectories, dim=1)
        ens_mse_batch = metrics.mse(
            ens_mean,
            target_states,
            None,
            mask=self.loss_mask,
            sum_vars=False,
        )

        return (
            trajectories,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        )

    def validation_step(self, batch, *args):
        """
        Run validation on single batch
        """
        super().validation_step(batch, *args)
        batch_idx = args[0]

        # Run ensemble forecast
        (
            prior_trajectories,
            _,
            spread_squared_batch,
            ens_mse_batch,
        ) = self.ensemble_step(batch)
        self.val_metrics["spread_squared"].append(spread_squared_batch)
        self.val_metrics["ens_mse"].append(ens_mse_batch)

        # Plot some example predictions using prior and encoder
        if (
            self.trainer.is_global_zero
            and batch_idx == 0
            and self.n_example_pred > 0
        ):
            # Roll out trajectories using variational distribution (encoder)
            (
                init_states,
                target_states,
                forcing_features,
                boundary_forcing,
                atmosphere_forcing,
                _,
            ) = batch

            # Sample trajectories using variational dist. for latent var.
            enc_trajectories, _ = self.sample_trajectories(
                init_states,
                forcing_features,
                boundary_forcing,
                atmosphere_forcing,
                target_states,
                self.ensemble_size,
                use_encoder=True,
            )

            # For now use val_steps_to_log to determine which steps
            # to make these plots for
            plot_log_steps = list(
                filter(
                    lambda s: s <= target_states.shape[1],
                    self.args.val_steps_to_log,
                )
            )

            # Plot forecasts, prior samples
            prior_plots = self.plot_ensemble_examples(
                batch,
                n_examples=self.n_example_pred,
                prediction=prior_trajectories,
                time_steps=plot_log_steps,
                log=False,
            )
            # Plot forecasts, variational samples
            vi_plots = self.plot_ensemble_examples(
                batch,
                n_examples=self.n_example_pred,
                prediction=enc_trajectories,
                time_steps=plot_log_steps,
                log=False,
            )

            # Store plots
            log_plot_dict = {}
            log_plot_dict.update(
                {
                    f"prior_{plot_key}": plot
                    for plot_key, plot in prior_plots.items()
                }
            )
            log_plot_dict.update(
                {f"vi_{plot_key}": plot for plot_key, plot in vi_plots.items()}
            )

            # Sample latent variable and plot
            # embed all features
            prev_state_0 = init_states[:, -1]
            prev_prev_state_0 = (
                init_states[:, 0] if init_states.shape[1] > 1 else None
            )
            grid_prev_emb, _, graph_emb = self.embedd_all(
                prev_state_0,
                prev_prev_state_0,
                forcing_features[:, 0],
                boundary_forcing[:, 0],
                atmosphere_forcing[:, 0],
            )  # (B, num_grid_nodes, d_h)
            # embed also including current grid state, for encoder
            grid_current_emb = self.embedd_current(
                prev_state_0,
                prev_prev_state_0,
                forcing_features[:, 0],
                boundary_forcing[:, 0],
                atmosphere_forcing[:, 0],
                target_states[:, 0],
            )  # (B, num_grid_nodes, d_h)

            # Create latent variable samples
            prior_dist = self.prior_model(
                grid_prev_emb, graph_emb=graph_emb
            )  # Gaussian, (B, num_mesh_nodes, d_latent)
            prior_samples = prior_dist.rsample(
                (self.num_latents_plot,)
            ).transpose(
                0, 1
            )  # (B, samples, num_mesh_nodes, d_latent)

            vi_dist = self.encoder(
                grid_current_emb, graph_emb=graph_emb
            )  # Gaussian, (B, num_mesh_nodes, d_latent)
            vi_samples = vi_dist.rsample((self.num_latents_plot,)).transpose(
                0, 1
            )  # (B, samples, num_mesh_nodes, d_latent)

            # Make plot for each example
            for example_i, (prior_ex_samples, vi_ex_samples) in enumerate(
                zip(prior_samples, vi_samples), start=1
            ):
                log_plot_dict[f"latent_samples_ex{example_i}"] = (
                    vis.plot_latent_samples(prior_ex_samples, vi_ex_samples)
                )

            if not self.trainer.sanity_checking:
                # Log all plots to wandb
                wandb.log(log_plot_dict)

            plt.close("all")

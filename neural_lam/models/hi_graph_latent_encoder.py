# Third-party
from torch import nn

# First-party
from neural_lam import utils
from neural_lam.interaction_net import FlexibleNet, PropagationNet
from neural_lam.models.base_latent_encoder import BaseLatentEncoder


class HiGraphLatentEncoder(BaseLatentEncoder):
    """
    Encoder that maps from grid to mesh and defines a latent distribution
    on mesh.
    Uses a hierarchical mesh graph.
    """

    def __init__(
        self,
        latent_dim,
        g2m_edge_index,
        m2m_edge_index,
        mesh_up_edge_index,
        hidden_dim,
        hidden_dim_grid,
        hidden_dim_mesh_nodes,
        hidden_dim_edge,
        intra_level_layers,
        hidden_layers=1,
        output_dist="isotropic",
        num_grid_con_mesh_nodes=None,
    ):
        super().__init__(
            latent_dim,
            output_dist,
        )

        print(
            f"HiGraphLatentEncoder hidden_dim: {hidden_dim}, "
            f"hidden_dim_grid: {hidden_dim_grid}"
        )
        # GNN from grid to mesh
        self.g2m_gnn = FlexibleNet(
            edge_index=g2m_edge_index,
            send_node_dim=hidden_dim_grid,
            rec_node_dim=hidden_dim_mesh_nodes,
            edge_dim=hidden_dim_edge,
            hidden_layers=hidden_layers,
            num_rec=num_grid_con_mesh_nodes,
            propagation=True,
            aggr="mean",
        )

        # GNNs going up through mesh levels
        self.mesh_up_gnns = nn.ModuleList(
            [
                PropagationNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )
                for edge_index in mesh_up_edge_index
            ]
        )

        # GNNs applied on intra-level in-between upwards propagation
        # Identity mappings if intra_level_layers = 0
        self.intra_level_gnns = nn.ModuleList(
            [
                utils.make_gnn_seq(
                    edge_index, intra_level_layers, hidden_layers, hidden_dim
                )
                for edge_index in m2m_edge_index
            ]
        )

        self.pre_mesh_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim_grid, hidden_dim)
        )

        # Final map to parameters
        self.latent_param_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1) + [self.output_dim],
            layer_norm=False,
        )

    # pylint: disable-next=arguments-differ
    def compute_dist_params(self, grid_rep, graph_emb, **kwargs):
        """
        Compute parameters of distribution over latent variable using the
        grid representation

        grid_rep: (B, N_grid, d_h)
        graph_emb: dict with graph embedding vectors, entries at least
            mesh: list of (B, N_mesh, d_h)
            g2m: (B, M_g2m, d_h)
            m2m: (B, M_g2m, d_h)
            mesh_up: list of (B, N_mesh, d_h)

        Returns:
        parameters: (B, num_mesh_nodes, d_output)
        """
        parallel = getattr(self, "graph_parallel", None)
        bottom_mesh = graph_emb["mesh"][0]
        if parallel is not None:
            bottom_mesh = parallel.replicate(bottom_mesh, 0)
        current_mesh_rep = self.g2m_gnn(
            grid_rep, bottom_mesh, graph_emb["g2m"]
        )  # (B, N_mesh, d_h)
        if parallel is not None:
            current_mesh_rep = parallel.select(current_mesh_rep, 0)

        current_mesh_rep = self.pre_mesh_proj(current_mesh_rep)

        # Run same level processing on level 0
        current_mesh_rep, _ = self.intra_level_gnns[0](
            current_mesh_rep, graph_emb["m2m"][0]
        )

        # Do not need to keep track of old edge or mesh reps here
        # Go from mesh level 1 to L
        for (
            up_gnn,
            intra_gnn_seq,
            mesh_up_level_rep,
            m2m_level_rep,
            mesh_level_rep,
        ) in zip(
            self.mesh_up_gnns,
            self.intra_level_gnns[1:],
            graph_emb["mesh_up"],
            graph_emb["m2m"][1:],
            graph_emb["mesh"][1:],
        ):
            # Apply up GNN
            new_node_rep = up_gnn(
                current_mesh_rep, mesh_level_rep, mesh_up_level_rep
            )  # (B, N_mesh[l], d_h)

            # Run same level processing on level l
            current_mesh_rep, _ = intra_gnn_seq(
                new_node_rep, m2m_level_rep
            )  # (B, N_mesh[l], d_h)

        # At final mesh level, map to parameter dim
        parameters = self.latent_param_map(current_mesh_rep)
        if parallel is not None:
            parameters = parallel.replicate(parameters, -1)
        return parameters  # (B, N_mesh[L], d_output)

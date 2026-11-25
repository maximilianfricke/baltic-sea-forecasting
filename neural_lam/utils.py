# Standard library
import contextlib
import os
import shutil
import warnings

# Third-party
import cartopy.crs as ccrs
import torch
import torch_geometric as pyg
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import nn
from tueplots import bundles, figsizes

# Local
from . import interaction_net
from .custom_loggers import CustomMLFlowLogger, CustomWandbLogger


class BufferList(nn.Module):
    """
    A list of torch buffer tensors that sit together as a Module with no
    parameters and only buffers.

    This should be replaced by a native torch BufferList once implemented.
    See: https://github.com/pytorch/pytorch/issues/37386
    """

    def __init__(self, buffer_tensors, persistent=True):
        super().__init__()
        self.n_buffers = len(buffer_tensors)
        for buffer_i, tensor in enumerate(buffer_tensors):
            self.register_buffer(f"b{buffer_i}", tensor, persistent=persistent)

    def __getitem__(self, key):
        return getattr(self, f"b{key}")

    def __len__(self):
        return self.n_buffers

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __itruediv__(self, other):
        """Divide each element in list with other"""
        return self.__imul__(1.0 / other)

    def __imul__(self, other):
        """Multiply each element in list with other"""
        for buffer_tensor in self:
            buffer_tensor *= other

        return self


def load_graph(graph_dir_path, datastore, device="cpu"):
    """Load all tensors representing the graph from `graph_dir_path`.

    Needs the following files for all graphs:
    - m2m_edge_index.pt
    - g2m_edge_index.pt
    - m2g_edge_index.pt
    - m2m_features.pt
    - g2m_features.pt
    - m2g_features.pt
    - mesh_features.pt

    And in addition for hierarchical graphs:
    - mesh_up_edge_index.pt
    - mesh_down_edge_index.pt
    - mesh_up_features.pt
    - mesh_down_features.pt

    Parameters
    ----------
    graph_dir_path : str
        Path to directory containing the graph files.
    device : str
        Device to load tensors to.

    Returns
    -------
    hierarchical : bool
        Whether the graph is hierarchical.
    graph : dict
        Dictionary containing the graph tensors, with keys as follows:
        - g2m_edge_index
        - m2g_edge_index
        - m2m_edge_index
        - mesh_up_edge_index
        - mesh_down_edge_index
        - g2m_features
        - m2g_features
        - m2m_node_features
        - mesh_up_features
        - mesh_down_features
        - mesh_static_features

    Load all tensors representing the graph
    """

    def loads_file(fn):
        return torch.load(
            os.path.join(graph_dir_path, fn),
            map_location=device,
            weights_only=True,
        )

    # Need to reindex some edge index to start from 0
    def reindex_func(edge_index):
        """
        Make both sender and receiver indices of edge_index start at 0
        """
        if edge_index.numel() == 0 or edge_index.shape[1] == 0:
            return edge_index
        return edge_index - edge_index.min(dim=1, keepdim=True)[0]

    # Load static node features
    mesh_pos = loads_file(
        "mesh_features.pt"
    )  # List of (N_mesh[l], d_mesh_static)
    # Static mesh features are normalized mesh node positions
    mesh_pos_min = (
        torch.stack(
            [level_pos.min(dim=0).values for level_pos in mesh_pos], dim=0
        )
        .min(dim=0)
        .values
    )
    mesh_pos_max = (
        torch.stack(
            [level_pos.max(dim=0).values for level_pos in mesh_pos], dim=0
        )
        .max(dim=0)
        .values
    )
    mesh_static_features = [
        (level_pos - mesh_pos_min) / (mesh_pos_max - mesh_pos_min)
        for level_pos in mesh_pos
    ]

    # Load edges (edge_index)
    m2m_edge_index = BufferList(
        [reindex_func(ei) for ei in loads_file("m2m_edge_index.pt")],
        persistent=False,
    )  # List of (2, M_m2m[l])
    g2m_edge_index = loads_file("g2m_edge_index.pt")  # (2, M_g2m)
    m2g_edge_index = loads_file("m2g_edge_index.pt")  # (2, M_m2g)

    # Change first indices to 0
    # m2g and g2m has to be handled specially as not all mesh nodes
    # might be indexed
    m2g_min_indices = m2g_edge_index.min(dim=1, keepdim=True)[0]

    # Zero-index g2m and m2g edge_index
    if m2g_min_indices[0] < m2g_min_indices[1]:
        # mesh has the first indices
        num_mesh_nodes = mesh_static_features[0].shape[0]

        m2g_edge_index = torch.stack(
            (
                m2g_edge_index[0],
                m2g_edge_index[1] - num_mesh_nodes,
            ),
            dim=0,
        )
        g2m_edge_index = torch.stack(
            (
                g2m_edge_index[0] - num_mesh_nodes,
                g2m_edge_index[1],
            ),
            dim=0,
        )
    else:
        # grid (interior) has the first indices
        # NOTE: Below works, but would be good with a better way to get this
        num_interior_nodes = m2g_edge_index[1].max() + 1
        num_grid_nodes = g2m_edge_index[0].max() + 1

        m2g_edge_index = torch.stack(
            (
                m2g_edge_index[0] - num_interior_nodes,
                m2g_edge_index[1],
            ),
            dim=0,
        )
        g2m_edge_index = torch.stack(
            (
                g2m_edge_index[0],
                g2m_edge_index[1] - num_grid_nodes,
            ),
            dim=0,
        )
    assert m2g_edge_index.min() >= 0, "Negative node index in m2g"
    assert g2m_edge_index.min() >= 0, "Negative node index in g2m"

    n_levels = len(m2m_edge_index)
    hierarchical = n_levels > 1  # Nor just single level mesh graph

    # Load static edge features
    # List of (M_m2m[l], d_edge_f)
    m2m_features = loads_file("m2m_features.pt")
    g2m_features = loads_file("g2m_features.pt")  # (M_g2m, d_edge_f)
    m2g_features = loads_file("m2g_features.pt")  # (M_m2g, d_edge_f)

    # Normalize by dividing with longest edge (found in m2m)
    # Skip empty levels (e.g. global cluster level with all edges filtered)
    longest_edge = max(
        (
            torch.max(level_features[:, 0]).item()
            for level_features in m2m_features
            if level_features.shape[0] > 0
        ),
        default=1.0,
    )

    m2m_features = BufferList(m2m_features, persistent=False)
    m2m_features /= longest_edge
    g2m_features = g2m_features / longest_edge
    m2g_features = m2g_features / longest_edge

    # Some checks for consistency
    assert (
        len(m2m_features) == n_levels
    ), "Inconsistent number of levels in mesh"
    assert (
        len(mesh_static_features) == n_levels
    ), "Inconsistent number of levels in mesh"

    # mesh_pos columns are (lon, lat); transform to PlateCarree gives (lon, lat)
    mesh_lon_lat = [
        torch.tensor(
            ccrs.PlateCarree().transform_points(
                datastore.coords_projection,
                mesh_coords[:, 0].numpy(),
                mesh_coords[:, 1].numpy(),
            )[
                :, :2
            ],  # Keep only 2d
            dtype=torch.float32,
        )
        for mesh_coords in mesh_pos
    ]

    if hierarchical:
        # Load up and down edges and features
        mesh_up_edge_index = BufferList(
            [reindex_func(ei) for ei in loads_file("mesh_up_edge_index.pt")],
            persistent=False,
        )  # List of (2, M_up[l])
        mesh_down_edge_index = BufferList(
            [reindex_func(ei) for ei in loads_file("mesh_down_edge_index.pt")],
            persistent=False,
        )  # List of (2, M_down[l])

        mesh_up_features = loads_file(
            "mesh_up_features.pt"
        )  # List of (M_up[l], d_edge_f)
        mesh_down_features = loads_file(
            "mesh_down_features.pt"
        )  # List of (M_down[l], d_edge_f)

        # Rescale
        mesh_up_features = BufferList(mesh_up_features, persistent=False)
        mesh_up_features /= longest_edge
        mesh_down_features = BufferList(mesh_down_features, persistent=False)
        mesh_down_features /= longest_edge

        mesh_static_features = BufferList(
            mesh_static_features, persistent=False
        )
    else:
        # Extract single mesh level
        m2m_edge_index = m2m_edge_index[0]
        m2m_features = m2m_features[0]
        mesh_static_features = mesh_static_features[0]

        (
            mesh_up_edge_index,
            mesh_down_edge_index,
            mesh_up_features,
            mesh_down_features,
        ) = ([], [], [], [])

    return hierarchical, {
        "g2m_edge_index": g2m_edge_index,
        "m2g_edge_index": m2g_edge_index,
        "m2m_edge_index": m2m_edge_index,
        "mesh_up_edge_index": mesh_up_edge_index,
        "mesh_down_edge_index": mesh_down_edge_index,
        "g2m_features": g2m_features,
        "m2g_features": m2g_features,
        "m2m_features": m2m_features,
        "mesh_up_features": mesh_up_features,
        "mesh_down_features": mesh_down_features,
        "mesh_static_features": mesh_static_features,
        "mesh_lon_lat": mesh_lon_lat,
    }


class ConditionalLayerNorm(nn.Module):
    """
    LayerNorm where the affine scale and offset are predicted from a
    conditioning embedding instead of being free parameters.
    """

    def __init__(self, normalized_shape, cond_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            normalized_shape, elementwise_affine=False
        )
        self.scale_layer = nn.Linear(cond_dim, normalized_shape)
        self.offset_layer = nn.Linear(cond_dim, normalized_shape)

    def forward(self, x, emb):
        """
        x: (..., normalized_shape)
        emb: (..., cond_dim), conditioning embedding, broadcast against x
        """
        scale = self.scale_layer(emb)  # (..., normalized_shape)
        offset = self.offset_layer(emb)  # (..., normalized_shape)
        return self.layer_norm(x) * scale + offset


def _mlp_layers(blueprint):
    """
    Build the Linear/SiLU stack of an MLP from a list blueprint.
    """
    hidden_layers = len(blueprint) - 2
    assert hidden_layers >= 0, "Invalid MLP blueprint"

    layers = []
    for layer_i, (dim1, dim2) in enumerate(zip(blueprint[:-1], blueprint[1:])):
        layers.append(nn.Linear(dim1, dim2))
        if layer_i != hidden_layers:
            layers.append(nn.SiLU())  # Swish activation

    return layers


# Rows (nodes or edges) per chunk for chunked inference, None = off. Only
# used while gradients are disabled, so training is never affected.
_INFERENCE_CHUNK_ROWS = None


@contextlib.contextmanager
def inference_chunking(rows):
    """Evaluate MLPs and message passing in chunks of ``rows`` rows.

    Inside this context and with gradients disabled, row-wise MLPs process
    their input ``rows`` rows at a time and InteractionNet, PropagationNet and
    FlexibleNet compute messages ``rows`` edges at a time, adding each chunk
    into the receiver aggregate. Results equal the unchunked evaluation up to
    floating-point summation order; peak memory falls from the size of the
    largest edge set to one chunk.
    """
    global _INFERENCE_CHUNK_ROWS
    previous = _INFERENCE_CHUNK_ROWS
    _INFERENCE_CHUNK_ROWS = rows
    try:
        yield
    finally:
        _INFERENCE_CHUNK_ROWS = previous


def inference_chunk_rows():
    """Chunk size if chunked inference is active and gradients are off."""
    if _INFERENCE_CHUNK_ROWS is None or torch.is_grad_enabled():
        return None
    return _INFERENCE_CHUNK_ROWS


def apply_rowwise_chunked(fn, x, rows):
    """``fn`` applied to ``x`` (..., N, F) in chunks of ``rows`` rows along
    dim -2; ``fn`` must act on each row independently."""
    n = x.shape[-2]
    out = None
    for start in range(0, n, rows):
        y = fn(x[..., start : start + rows, :])
        if out is None:
            out = y.new_empty(y.shape[:-2] + (n,) + y.shape[-1:])
        out[..., start : start + rows, :] = y
    return out


class MLP(nn.Sequential):
    """
    Plain MLP, optionally with a LayerNorm on the output.

    Subclasses nn.Sequential so that parameters keep their integer-indexed
    names ("0.weight", "2.weight", ...), and accepts (and ignores) a
    conditioning embedding so that it is call-compatible with CondMLP.
    """

    def __init__(self, blueprint, layer_norm=True):
        layers = _mlp_layers(blueprint)
        if layer_norm:
            layers.append(nn.LayerNorm(blueprint[-1]))
        super().__init__(*layers)

    # pylint: disable-next=arguments-differ
    def forward(self, x, emb=None):
        """
        x: (..., blueprint[0])
        emb: unused, present for call-compatibility with CondMLP
        """
        rows = inference_chunk_rows()
        if rows is not None and x.dim() >= 2 and x.shape[-2] > rows:
            return apply_rowwise_chunked(
                lambda part: nn.Sequential.forward(self, part), x, rows
            )
        return super().forward(x)


class CondMLP(nn.Module):
    """
    MLP with a ConditionalLayerNorm on the output, modulated by a
    conditioning embedding.
    """

    def __init__(self, blueprint, layer_norm=True, cond_dim=None):
        super().__init__()
        assert cond_dim is not None, "CondMLP requires a cond_dim"

        self.mlp_layers = nn.Sequential(*_mlp_layers(blueprint))

        if layer_norm:
            self.layer_norm = ConditionalLayerNorm(blueprint[-1], cond_dim)
        else:
            self.layer_norm = None

    def forward(self, x, emb=None):
        """
        x: (..., blueprint[0])
        emb: (..., cond_dim), conditioning embedding
        """
        rows = inference_chunk_rows()
        # Chunk only if emb is shared by all rows (no row dimension)
        shared_emb = emb is None or emb.dim() < x.dim() or emb.shape[-2] == 1
        if (
            rows is not None
            and shared_emb
            and x.dim() >= 2
            and x.shape[-2] > rows
        ):
            return apply_rowwise_chunked(
                lambda part: self._forward(part, emb), x, rows
            )
        return self._forward(x, emb)

    def _forward(self, x, emb):
        x = self.mlp_layers(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x, emb)

        return x


def make_mlp(blueprint, layer_norm=True, cond_dim=None):
    """
    Create MLP from list blueprint, with
    input dimensionality: blueprint[0]
    output dimensionality: blueprint[-1] and
    hidden layers of dimensions: blueprint[1], ..., blueprint[-2]

    if layer_norm is True, includes a LayerNorm layer at
    the output (as used in GraphCast)

    if cond_dim is not None, that output LayerNorm is a ConditionalLayerNorm
    whose scale and offset are predicted from a conditioning embedding of
    dimensionality cond_dim, passed as the second argument to forward.

    The returned module is always called as mlp(x, emb=None).
    """
    if cond_dim is None:
        return MLP(blueprint, layer_norm=layer_norm)

    return CondMLP(blueprint, layer_norm=layer_norm, cond_dim=cond_dim)


def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of
    the page width.
    """
    # If latex is not available, some visualizations might not render
    # correctly, but will at least not raise an error. Alternatively, use
    # unicode raised numbers.
    usetex = True if shutil.which("latex") else False
    bundle = bundles.neurips2023(usetex=usetex, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] / fraction,
        original_figsize[1],
    )
    return bundle


@rank_zero_only
def rank_zero_print(*args, **kwargs):
    """Print only from rank 0 process"""
    print(*args, **kwargs)


def init_training_logger_metrics(training_logger, val_steps):
    """
    Set up logger metrics to track
    """
    experiment = training_logger.experiment
    if isinstance(training_logger, WandbLogger):
        experiment.define_metric("val_mean_loss", summary="min")
        for step in val_steps:
            experiment.define_metric(f"val_loss_unroll{step}", summary="min")
    elif isinstance(training_logger, MLFlowLogger):
        pass
    else:
        warnings.warn(
            "Only WandbLogger & MLFlowLogger is supported for tracking metrics.\
             Experiment results will only go to stdout."
        )


@rank_zero_only
def setup_training_logger(datastore, args, run_name):
    """

    Parameters
    ----------
    datastore : Datastore
        Datastore object.

    args : argparse.Namespace
        Arguments from command line.

    run_name : str
        Name of the run.

    Returns
    -------
    logger : pytorch_lightning.loggers.base
        Logger object.
    """

    if args.logger == "wandb":
        logger = CustomWandbLogger(
            project=args.logger_project,
            name=run_name,
            config=dict(training=vars(args), datastore=datastore._config),
        )
    elif args.logger == "mlflow":
        url = os.getenv("MLFLOW_TRACKING_URI")
        if url is None:
            raise ValueError(
                "MLFlow logger requires setting MLFLOW_TRACKING_URI in env."
            )
        logger = CustomMLFlowLogger(
            experiment_name=args.logger_project,
            tracking_uri=url,
            run_name=run_name,
        )
        logger.log_hyperparams(
            dict(training=vars(args), datastore=datastore._config)
        )

    return logger


def inverse_softplus(x, beta=1, threshold=20):
    """
    Inverse of torch.nn.functional.softplus

    Input is clamped to approximately positive values of x, and the function is
    linear for inputs above x*beta for numerical stability.

    Note that this torch.clamp will make gradients 0, but this is not a
    problem as values of x that are this close to 0 have gradients of 0 anyhow.
    """
    x_clamped = torch.clamp(
        x, min=torch.log(torch.tensor(1e-6 + 1)) / beta, max=threshold / beta
    )

    non_linear_part = torch.log(torch.expm1(x_clamped * beta)) / beta

    below_threshold = x * beta <= threshold

    x = torch.where(condition=below_threshold, input=non_linear_part, other=x)

    return x


def inverse_sigmoid(x):
    """
    Inverse of torch.sigmoid

    Sigmoid output takes values in [0,1], this makes sure input is just within
    this interval.
    Note that this torch.clamp will make gradients 0, but this is not a problem
    as values of x that are this close to 0 or 1 have gradients of 0 anyhow.
    """
    x_clamped = torch.clamp(x, min=1e-6, max=1 - 1e-6)
    return torch.log(x_clamped / (1 - x_clamped))


class IdentityModule(nn.Module):
    """
    A identity operator that can return multiple inputs
    """

    def forward(self, *args):
        """Return input args"""
        return args


def make_gnn_seq(edge_index, num_gnn_layers, hidden_layers, hidden_dim):
    """
    Make a sequential GNN module propagating both node and edge representations
    """
    if num_gnn_layers == 0:
        # If no layers, return identity
        return IdentityModule()
    return pyg.nn.Sequential(
        "mesh_rep, edge_rep",
        [
            (
                interaction_net.InteractionNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                ),
                "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep",
            )
            for _ in range(num_gnn_layers)
        ],
    )

"""Load a saved global graph into a flat list of node layers and edge sets."""

# Standard library
import os
from dataclasses import dataclass
from typing import List

# Third-party
import numpy as np
import torch

# Local
from .. import utils
from ..config import load_config_and_datastores


@dataclass
class Layer:
    """One set of nodes: the grid or one mesh level."""

    name: str
    lon_lat: np.ndarray  # (N, 2) degrees
    level: int  # -1 for the grid, mesh level otherwise

    @property
    def xyz(self) -> np.ndarray:
        """(N, 3) positions on the unit sphere."""
        return lon_lat_to_xyz(self.lon_lat)

    def __len__(self):
        return len(self.lon_lat)


@dataclass
class EdgeSet:
    """Directed edges from nodes of layer ``src`` to nodes of layer ``dst``.

    ``edge_index`` holds layer-local indices: row 0 indexes ``src``, row 1
    indexes ``dst``.
    """

    name: str
    kind: str  # g2m, m2g, m2m, up or down
    src: int  # index into ViewerGraph.layers
    dst: int
    edge_index: np.ndarray  # (2, E)

    @property
    def n_edges(self):
        return self.edge_index.shape[1]


@dataclass
class ViewerGraph:
    name: str
    hierarchical: bool
    layers: List[Layer]  # grid first, then mesh levels bottom to top
    edge_sets: List[EdgeSet]


def lon_lat_to_xyz(lon_lat: np.ndarray) -> np.ndarray:
    lon, lat = np.deg2rad(lon_lat[:, 0]), np.deg2rad(lon_lat[:, 1])
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        axis=1,
    )


def _mesh_features_to_lon_lat(features: np.ndarray) -> np.ndarray:
    """Invert (sin_lon, cos_lon, sin_lat, cos_lat, ...) to degrees."""
    lon = np.arctan2(features[:, 0], features[:, 1])
    lat = np.arctan2(features[:, 2], features[:, 3])
    return np.rad2deg(np.stack([lon, lat], axis=1))


def load_viewer_graph(config_path: str, graph_name: str) -> ViewerGraph:
    """Load graph ``graph_name`` of the datastore in ``config_path``."""
    _, datastore, *_ = load_config_and_datastores(config_path=config_path)
    graph_dir_path = os.path.join(datastore.root_path, "graphs", graph_name)
    if not os.path.isdir(graph_dir_path):
        raise FileNotFoundError(f"Graph directory not found: {graph_dir_path}")

    # load_graph makes all edge indices layer-local
    hierarchical, graph = utils.load_graph(
        graph_dir_path=graph_dir_path, datastore=datastore
    )

    # Global graphs are built on the interior (ocean) grid nodes only
    interior_mask = datastore.get_mask(surface=True, stacked=True, invert=False)
    grid_xy = datastore.get_xy("state", stacked=True)[interior_mask]
    layers = [Layer("Grid", np.asarray(grid_xy, dtype=np.float64), -1)]

    mesh_features = torch.load(
        os.path.join(graph_dir_path, "mesh_features.pt"),
        map_location="cpu",
        weights_only=True,
    )
    for level, features in enumerate(mesh_features):
        layers.append(
            Layer(
                f"Mesh {level}",
                _mesh_features_to_lon_lat(features.numpy().astype(np.float64)),
                level,
            )
        )

    def as_list(edge_index):
        if torch.is_tensor(edge_index):
            return [edge_index.numpy()]
        return [ei.numpy() for ei in edge_index]

    mesh = lambda level: level + 1  # noqa: E731, layer index of mesh level
    edge_sets = [
        EdgeSet("g2m", "g2m", 0, mesh(0), graph["g2m_edge_index"].numpy()),
        EdgeSet("m2g", "m2g", mesh(0), 0, graph["m2g_edge_index"].numpy()),
    ]
    for level, ei in enumerate(as_list(graph["m2m_edge_index"])):
        edge_sets.append(
            EdgeSet(f"m2m {level}", "m2m", mesh(level), mesh(level), ei)
        )
    if hierarchical:
        for level, ei in enumerate(as_list(graph["mesh_up_edge_index"])):
            edge_sets.append(
                EdgeSet(
                    f"up {level}->{level + 1}",
                    "up",
                    mesh(level),
                    mesh(level + 1),
                    ei,
                )
            )
        for level, ei in enumerate(as_list(graph["mesh_down_edge_index"])):
            edge_sets.append(
                EdgeSet(
                    f"down {level + 1}->{level}",
                    "down",
                    mesh(level + 1),
                    mesh(level),
                    ei,
                )
            )

    for es in edge_sets:
        if es.n_edges == 0:
            continue
        assert es.edge_index[0].max() < len(
            layers[es.src]
        ), f"{es.name}: sender index out of range for {layers[es.src].name}"
        assert es.edge_index[1].max() < len(
            layers[es.dst]
        ), f"{es.name}: receiver index out of range for {layers[es.dst].name}"

    return ViewerGraph(graph_name, hierarchical, layers, edge_sets)

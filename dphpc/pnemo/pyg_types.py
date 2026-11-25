"""PyG imports without importing PhysicsNeMo's unrelated model dependencies."""

from torch_geometric.data import Data as PyGData, HeteroData

PYG_AVAILABLE = True
GraphType = PyGData | HeteroData

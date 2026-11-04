"""Relative memory and communication cost of a partitioned graph.

Only terms that differ between partitionings of the same graph over the same
number of partitions are kept. Fixed memory (weights, optimizer state,
checkpoint inputs), hidden_dim, bytes per value, the number of unrolled steps
and forward/backward factors are dropped.

Every edge is owned by the partition of its receiver node. For partition p:

    M(p) = sum_l n(l) N(p, l) + sum_s m(s) E(p, s) + sum_s u(s) H(p, s)
    C(p) = sum_s u(s) H(p, s)

with N(p, l) nodes of layer l, E(p, s) edges of edge set s and H(p, s) halo
nodes: distinct senders outside p with an edge of s into p. n(l) and m(s) are
the values stored per node and per edge during one forecast step (the
gradient-checkpointed segment), in units of hidden_dim, and u(s) the number of
times edge set s is applied per step. Together they form a cost model, which
``measure_weights.py`` measures for one model configuration and saves as one
JSON file with a "memory" section (n, m) and a "usage" section (u).
"""

# Standard library
import json
from dataclasses import dataclass, field
from typing import Dict, List

# Third-party
import numpy as np

# Local
from .graph import ViewerGraph

MEMORY_UNIT = (
    "values stored per item / hidden_dim, one checkpointed forecast step"
)
USAGE_UNIT = "applications per forecast step"


@dataclass
class CostModel:
    name: str
    node: Dict[str, float]  # memory: layer name -> n(l)
    edge: Dict[str, float]  # memory: edge set name -> m(s)
    uses: Dict[str, float]  # usage: edge set name -> u(s)
    source: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    total_per_step_values: float = 0.0  # measured total, for checking
    measured: bool = True

    @classmethod
    def uniform(cls, graph: ViewerGraph):
        """Every node, edge and use counts once (no measurement)."""
        return cls(
            "Uniform (unmeasured)",
            {layer.name: 1.0 for layer in graph.layers},
            {es.name: 1.0 for es in graph.edge_sets},
            {es.name: 1.0 for es in graph.edge_sets},
            measured=False,
        )

    @classmethod
    def load(cls, path: str):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        memory, usage = d["memory"], d["usage"]
        return cls(
            d["name"],
            memory["node"],
            memory["edge"],
            usage["edge"],
            d.get("source", {}),
            memory.get("notes", []),
            memory.get("total_per_step_values", 0.0),
        )

    def save(self, path: str):
        d = {
            "name": self.name,
            "source": self.source,
            "memory": {
                "unit": MEMORY_UNIT,
                "node": self.node,
                "edge": self.edge,
                "total_per_step_values": self.total_per_step_values,
                "notes": self.notes,
            },
            "usage": {"unit": USAGE_UNIT, "edge": self.uses},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    def check(self, graph: ViewerGraph):
        missing = [
            layer.name for layer in graph.layers if layer.name not in self.node
        ] + [
            es.name
            for es in graph.edge_sets
            if es.name not in self.edge or es.name not in self.uses
        ]
        if missing:
            raise ValueError(
                f"Cost model {self.name!r} has no values for: {missing}"
            )


def node_weights(graph: ViewerGraph, w: CostModel) -> List[np.ndarray]:
    """Load of each node: its own storage plus that of the edges it receives.

    w(v) = n(l) + sum_s m(s) * (edges of s received by v)
    """
    weights = [
        np.full(len(layer), w.node[layer.name]) for layer in graph.layers
    ]
    for es in graph.edge_sets:
        weights[es.dst] += w.edge[es.name] * np.bincount(
            es.edge_index[1], minlength=len(graph.layers[es.dst])
        )
    return weights


def halo_counts(edge_index, src_parts, dst_parts, k) -> np.ndarray:
    """H(p, s) for every p: distinct remote senders with an edge into p."""
    cut = src_parts[edge_index[0]] != dst_parts[edge_index[1]]
    pairs = np.unique(
        np.stack([edge_index[0, cut], dst_parts[edge_index[1, cut]]]), axis=1
    )
    return np.bincount(pairs[1], minlength=k)


@dataclass
class PartitionCosts:
    k: int
    # Per-partition terms of M(p), each (k,)
    node_terms: Dict[str, np.ndarray] = field(default_factory=dict)
    edge_terms: Dict[str, np.ndarray] = field(default_factory=dict)
    halo_terms: Dict[str, np.ndarray] = field(default_factory=dict)
    neighbours: np.ndarray = None  # (k,) partitions sending to p
    single: float = 0.0  # M of the unpartitioned graph

    @property
    def memory(self) -> np.ndarray:
        """M(p)"""
        return (
            sum(self.node_terms.values())
            + sum(self.edge_terms.values())
            + sum(self.halo_terms.values())
        )

    @property
    def communication(self) -> np.ndarray:
        """C(p)"""
        return sum(self.halo_terms.values())

    @property
    def busiest(self) -> int:
        return int(np.argmax(self.memory))

    @property
    def max_share(self) -> float:
        """max_p M(p) relative to the unpartitioned M (ideal 1 / k)."""
        return self.memory.max() / self.single

    @property
    def imbalance(self) -> float:
        """max_p M(p) / mean_p M(p) (ideal 1)."""
        return self.memory.max() / self.memory.mean()


def partition_costs(
    graph: ViewerGraph, parts: List[np.ndarray], k: int, w: CostModel
) -> PartitionCosts:
    costs = PartitionCosts(k)
    for layer, p in zip(graph.layers, parts):
        costs.node_terms[layer.name] = w.node[layer.name] * np.bincount(
            p, minlength=k
        )
    sends_to = np.zeros((k, k), dtype=bool)  # [sender part, receiver part]
    for es in graph.edge_sets:
        src_p, dst_p = parts[es.src], parts[es.dst]
        costs.edge_terms[es.name] = w.edge[es.name] * np.bincount(
            dst_p[es.edge_index[1]], minlength=k
        )
        costs.halo_terms[es.name] = w.uses[es.name] * halo_counts(
            es.edge_index, src_p, dst_p, k
        )
        sends_to[src_p[es.edge_index[0]], dst_p[es.edge_index[1]]] = True
    np.fill_diagonal(sends_to, False)
    costs.neighbours = sends_to.sum(axis=0)
    costs.single = sum(
        w.node[layer.name] * len(layer) for layer in graph.layers
    ) + sum(w.edge[es.name] * es.n_edges for es in graph.edge_sets)
    return costs

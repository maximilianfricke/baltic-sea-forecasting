# Third-party
import numpy as np

# First-party
from neural_lam.graph_viewer.costs import (
    CostModel,
    node_weights,
    partition_costs,
)
from neural_lam.graph_viewer.graph import EdgeSet, Layer, ViewerGraph


def _tiny_graph():
    """4 grid nodes, 2 mesh nodes.

    g2m: grid 0, 1 -> mesh 0; grid 2, 3 -> mesh 1; grid 1 -> mesh 1
    m2g: mesh 0 -> grid 0, 1, 2; mesh 1 -> grid 3
    """
    layers = [
        Layer("Grid", np.zeros((4, 2)), -1),
        Layer("Mesh 0", np.zeros((2, 2)), 0),
    ]
    edge_sets = [
        EdgeSet(
            "g2m", "g2m", 0, 1, np.array([[0, 1, 2, 3, 1], [0, 0, 1, 1, 1]])
        ),
        EdgeSet("m2g", "m2g", 1, 0, np.array([[0, 0, 0, 1], [0, 1, 2, 3]])),
    ]
    return ViewerGraph("tiny", False, layers, edge_sets)


WEIGHTS = CostModel(
    name="test",
    node={"Grid": 1.0, "Mesh 0": 10.0},
    edge={"g2m": 2.0, "m2g": 3.0},
    uses={"g2m": 5.0, "m2g": 7.0},
)


def test_node_weights_add_received_edges():
    w = node_weights(_tiny_graph(), WEIGHTS)
    # Grid: n + 3 * (m2g edges received, one each)
    np.testing.assert_allclose(w[0], [4, 4, 4, 4])
    # Mesh: n + 2 * (g2m edges received: 2 and 3)
    np.testing.assert_allclose(w[1], [14, 16])


def test_partition_costs_by_hand():
    graph = _tiny_graph()
    # Partition 0: grid 0, 1 and mesh 0. Partition 1: grid 2, 3 and mesh 1.
    parts = [np.array([0, 0, 1, 1]), np.array([0, 1])]
    costs = partition_costs(graph, parts, 2, WEIGHTS)

    np.testing.assert_allclose(costs.node_terms["Grid"], [2, 2])
    np.testing.assert_allclose(costs.node_terms["Mesh 0"], [10, 10])
    # Edges belong to the receiver's partition
    np.testing.assert_allclose(costs.edge_terms["g2m"], [2 * 2, 2 * 3])
    np.testing.assert_allclose(costs.edge_terms["m2g"], [3 * 2, 3 * 2])
    # Cut edges: g2m grid 1 -> mesh 1 (into p1); m2g mesh 0 -> grid 2 (p1)
    np.testing.assert_allclose(costs.halo_terms["g2m"], [0, 5])
    np.testing.assert_allclose(costs.halo_terms["m2g"], [0, 7])

    np.testing.assert_allclose(
        costs.memory, [2 + 10 + 4 + 6, 2 + 10 + 6 + 6 + 12]
    )
    np.testing.assert_allclose(costs.communication, [0, 12])
    np.testing.assert_array_equal(costs.neighbours, [0, 1])
    assert costs.single == 4 * 1 + 2 * 10 + 5 * 2 + 4 * 3
    assert costs.busiest == 1


def test_cost_model_round_trip(tmp_path):
    path = tmp_path / "model.json"
    model = CostModel(
        "test",
        WEIGHTS.node,
        WEIGHTS.edge,
        WEIGHTS.uses,
        source={"graph": "tiny"},
        notes=["a note"],
        total_per_step_values=123,
    )
    model.save(path)
    loaded = CostModel.load(path)
    assert loaded == model
    assert loaded.measured
    assert not CostModel.uniform(_tiny_graph()).measured

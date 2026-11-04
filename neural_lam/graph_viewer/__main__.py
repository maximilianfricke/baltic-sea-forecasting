"""Interactive 3D viewer of a global graph with simple partitioning.

Example:
    python -m neural_lam.graph_viewer \
        --config_path sample_data/global_ocean_1.yaml \
        --strategy rcb --k 4
"""

# Standard library
import os
from argparse import ArgumentParser

# Local
from .costs import CostModel
from .graph import load_viewer_graph
from .partition import STRATEGIES
from .viewer import NO_PARTITION, run


def main(input_args=None):
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        "--graph_name",
        type=str,
        default="global_cluster_graph",
        help="Graph to show (default: the graph of the realistic baseline)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=NO_PARTITION,
        choices=[NO_PARTITION] + list(STRATEGIES),
        help="Initial partitioning strategy (can be changed in the viewer)",
    )
    parser.add_argument(
        "--k", type=int, default=4, help="Initial number of partitions"
    )
    parser.add_argument(
        "--cost_model",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "weights", "realistic_baseline.json"
        ),
        help="Cost model JSON from measure_weights.py, or 'uniform' "
        "(default: the realistic Graph-EFM baseline)",
    )
    parser.add_argument(
        "--level_gap",
        type=float,
        default=0.1,
        help="Radial distance between the grid and each mesh level",
    )
    parser.add_argument(
        "--show",
        nargs="*",
        default=[],
        help="Structures to enable at start, e.g. 'g2m edges'",
    )
    parser.add_argument(
        "--hide",
        nargs="*",
        default=[],
        help="Structures to disable at start, e.g. 'Grid nodes'",
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        help="Render one frame to this image file and exit",
    )
    args = parser.parse_args(input_args)

    graph = load_viewer_graph(args.config_path, args.graph_name)
    cost_model = (
        CostModel.uniform(graph)
        if args.cost_model == "uniform"
        else CostModel.load(args.cost_model)
    )
    enable = {name: True for name in args.show}
    enable.update({name: False for name in args.hide})
    run(
        graph,
        level_gap=args.level_gap,
        strategy=args.strategy,
        k=args.k,
        screenshot=args.screenshot,
        enable=enable,
        cost_model=cost_model,
    )


if __name__ == "__main__":
    main()

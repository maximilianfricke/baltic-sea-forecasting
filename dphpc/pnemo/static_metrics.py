"""CPU-only communication sizes from the same PhysicsNeMo GraphPartitions."""

import argparse
import csv
from pathlib import Path

import torch

from .partitions import DEFAULT, METHODS, graph_data, load_owners
from dphpc.third_party.physicsnemo.distributed_graph import (
    edge_index_to_csc,
    partition_graph_with_id_mapping,
)


def partition_metrics(partition):
    rank = partition.partition_rank
    received = (
        sum(row[rank] for row in partition.sizes) - partition.sizes[rank][rank]
    )
    sent = sum(partition.sizes[rank]) - partition.sizes[rank][rank]
    return dict(
        halo_nodes_received=received,
        halo_nodes_sent=sent,
        bytes_received_per_exchange=received * 64 * 4 * 2,
        bytes_sent_per_exchange=sent * 64 * 4 * 2,
        local_edges=partition.num_local_indices,
        local_sources=partition.num_local_src_nodes,
        owned_sources=len(partition.map_partitioned_src_ids_to_global),
        owned_destinations=len(partition.map_partitioned_dst_ids_to_global),
    )


def write_metrics(output, K=None, root=DEFAULT, methods=METHODS):
    torch.set_num_threads(1)
    info = graph_data()
    rows = []
    for size in ((K,) if K else (2, 4)):
        for method in methods:
            owners, _ = load_owners(root, method, size, info)
            balance = {
                key: float(torch.bincount(value, minlength=size).max())
                / (len(value) / size)
                for key, value in owners.items()
            }
            for relation, (src, dst, edges) in info[2].items():
                offsets, indices, _ = edge_index_to_csc(edges, len(owners[dst]))
                for rank in range(size):
                    partition = partition_graph_with_id_mapping(
                        offsets,
                        indices,
                        owners[src],
                        owners[dst],
                        size,
                        rank,
                        torch.device("cpu"),
                    )
                    rows.append(
                        dict(
                            partitioner=method,
                            K=size,
                            rank=rank,
                            relation=relation,
                            **partition_metrics(partition),
                            **{
                                f"{key}_imbalance": value
                                for key, value in balance.items()
                            },
                            graph_fingerprint=info[-1],
                        )
                    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, choices=(2, 4))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_metrics(args.output, args.K)

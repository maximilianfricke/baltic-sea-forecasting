"""Graph identity, saved rank-file validation and ownership audit."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch

from neural_lam.config import load_config_and_datastores
from neural_lam.utils import load_graph

METHODS = ("chunks", "rcb", "metis", "mtkahypar_km1")
DEFAULT = Path(__file__).resolve().parents[2] / "partitions/1deg"
CONFIG = Path(
    "/scratch/users/nrottstegge/njord-data/sample_data/global_ocean_1.yaml"
)
GRAPH = "global_hierarchical_graph"


def graph_data(config=CONFIG, name=GRAPH):
    _, store, *_ = load_config_and_datastores(str(config))
    hierarchical, graph = load_graph(
        store.root_path / "graphs" / name, store, "cpu"
    )
    mesh = (
        list(graph["mesh_static_features"])
        if hierarchical
        else [graph["mesh_static_features"]]
    )
    sizes = {
        "grid": len(
            store.get_xy("state", True)[store.get_mask(True, True, False)]
        ),
        **{f"mesh_{i}": len(v) for i, v in enumerate(mesh)},
    }
    m2m = (
        list(graph["m2m_edge_index"])
        if hierarchical
        else [graph["m2m_edge_index"]]
    )
    relations = {"g2m": ("grid", "mesh_0", graph["g2m_edge_index"])}
    relations.update(
        {f"m2m_{i}": (f"mesh_{i}", f"mesh_{i}", ei) for i, ei in enumerate(m2m)}
    )
    for i, (up, down) in enumerate(
        zip(graph["mesh_up_edge_index"], graph["mesh_down_edge_index"])
    ):
        relations[f"up_{i}"] = (f"mesh_{i}", f"mesh_{i+1}", up)
        relations[f"down_{i}"] = (f"mesh_{i+1}", f"mesh_{i}", down)
    relations["m2g"] = ("mesh_0", "grid", graph["m2g_edge_index"])
    features = torch.load(
        store.root_path / "graphs" / name / "mesh_features.pt",
        weights_only=True,
    )
    xyz = []
    for v in features:
        slon, clon, slat, clat = v[:, :4].unbind(1)
        xyz.append(torch.stack((clat * clon, clat * slon, slat), 1))
    digest = hashlib.sha256()
    for t in [*xyz, *[r[2] for r in relations.values()]]:
        t = t.cpu().contiguous()
        digest.update(str((tuple(t.shape), t.dtype)).encode())
        digest.update(t.numpy().tobytes())
    return graph, sizes, relations, store, digest.hexdigest()


def load_owners(root, method, K, info):
    graph, sizes, relations, store, fingerprint = info
    owners = {
        key: torch.full((size,), -1, dtype=torch.long)
        for key, size in sizes.items()
    }
    shards = []
    xy = store.get_xy("state", True)[store.get_mask(True, True, False)]
    for rank in range(K):
        shard = torch.load(
            Path(root) / method / f"k{K}" / f"rank{rank}.pt",
            weights_only=True,
            mmap=True,
        )
        assert (shard["partitioner"], shard["K"], shard["rank"]) == (
            method,
            K,
            rank,
        )
        assert (
            shard["graph_fingerprint"] == fingerprint
        ), "Saved graph/node-order identity differs"
        assert set(shard["node_ids"]) == set(sizes)
        for key, ids in shard["node_ids"].items():
            assert ids.dtype == torch.long and ids.ndim == 1
            assert len(ids) > 0, f"{method} K={K} rank={rank}: empty {key}"
            assert bool(((ids >= 0) & (ids < sizes[key])).all())
            assert len(torch.unique(ids)) == len(ids) and torch.equal(
                ids, ids.sort().values
            )
            assert bool((owners[key][ids] == -1).all()), "Duplicate ownership"
            owners[key][ids] = rank
        grid = shard["node_ids"]["grid"]
        np.testing.assert_allclose(
            shard["grid_lon_lat_deg"], xy[grid], rtol=0, atol=1e-8
        )
        for values in shard["data"].values():
            assert values.shape[1] == len(grid)
        assert len(shard["land_mask"]) == len(grid)
        shards.append(shard)
    assert all(bool(((o >= 0) & (o < K)).all()) for o in owners.values())
    for rank, shard in enumerate(shards):
        for key, (src, dst, edge) in relations.items():
            expected = torch.where(owners[dst][edge[1]] == rank)[0]
            assert torch.equal(
                shard["relations"][key]["edge_ids"], expected
            ), f"{method}: {key} edge order mismatch"
    return owners, shards


def audit(root=DEFAULT, info=None):
    info = info or graph_data()
    table = []
    for method in METHODS:
        for K in (2, 4):
            owners, _ = load_owners(root, method, K, info)
            for key, owner in owners.items():
                counts = torch.bincount(owner, minlength=K).tolist()
                imbalance = max(counts) / (len(owner) / K)
                table.append(
                    dict(
                        partitioner=method,
                        K=K,
                        node_set=key,
                        counts="/".join(map(str, counts)),
                        max_mean=imbalance,
                        limit=1.05 if key == "grid" else 1.25,
                        status=(
                            "WARN"
                            if imbalance > (1.05 if key == "grid" else 1.25)
                            else "OK"
                        ),
                    )
                )
    with (Path(root) / "validation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    for row in table:
        print(
            f"{row['partitioner']:16} K={row['K']} {row['node_set']:6} {row['counts']:26} {row['max_mean']:.3f} {row['status']}",
            flush=True,
        )
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT)
    args = p.parse_args()
    torch.set_num_threads(1)
    audit(args.root)

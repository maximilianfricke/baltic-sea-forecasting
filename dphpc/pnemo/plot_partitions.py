"""CPU ownership maps and per-rank counts from the actual saved rank files."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np
import torch

from .partitions import DEFAULT, GRAPH, METHODS, graph_data, load_owners
from .static_metrics import partition_metrics
from dphpc.third_party.physicsnemo.distributed_graph import (
    edge_index_to_csc,
    partition_graph_with_id_mapping,
)

RANK_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd")
NAMES = dict(
    chunks="chunks", rcb="RCB", metis="METIS", mtkahypar_km1="Mt-KaHyPar km1"
)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"), dpi=200)
    plt.close(fig)


def project(lonlat, center):
    lon = np.deg2rad(lonlat[:, 0] - center)
    lat = np.deg2rad(lonlat[:, 1])
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    visible = np.cos(lat) * np.cos(lon) >= -1e-10
    return x, y, visible


def globe(ax, coordinates, owners, center, small=False):
    ax.add_patch(
        Circle(
            (0, 0),
            1,
            facecolor="#f3f4f6",
            edgecolor="#777777",
            linewidth=0.5,
            zorder=0,
        )
    )
    for lon in range(-180, 180, 30):
        points = np.column_stack((np.full(181, lon), np.linspace(-90, 90, 181)))
        x, y, visible = project(points, center)
        ax.plot(
            np.where(visible, x, np.nan),
            np.where(visible, y, np.nan),
            color="#c9ced4",
            lw=0.25,
            zorder=1,
        )
    for lat in range(-60, 90, 30):
        points = np.column_stack(
            (np.linspace(-180, 180, 361), np.full(361, lat))
        )
        x, y, visible = project(points, center)
        ax.plot(
            np.where(visible, x, np.nan),
            np.where(visible, y, np.nan),
            color="#c9ced4",
            lw=0.25,
            zorder=1,
        )
    for name in coordinates:
        size = 0.3 if name == "grid" else (2.5 if name == "mesh_0" else 8)
        if small:
            size *= 0.7
        x, y, visible = project(coordinates[name], center)
        ax.scatter(
            x[visible],
            y[visible],
            s=size,
            c=owners[name][visible],
            cmap=ListedColormap(RANK_COLORS),
            vmin=0,
            vmax=3,
            alpha=0.75 if name == "grid" else 0.95,
            edgecolors="none" if name != "mesh_1" else "#222222",
            linewidths=0 if name != "mesh_1" else 0.12,
            rasterized=True,
            zorder=2 if name == "grid" else (3 if name == "mesh_0" else 4),
        )
    ax.set(xlim=(-1.03, 1.03), ylim=(-1.03, 1.03), aspect="equal")
    ax.axis("off")


def legends(K=4):
    return [
        Patch(facecolor=RANK_COLORS[r], label=f"rank {r}") for r in range(K)
    ] + [
        Line2D(
            [],
            [],
            marker="o",
            color="gray",
            linestyle="",
            markersize=s,
            label=name,
        )
        for name, s in [("grid", 2), ("G0", 4), ("G1", 6)]
    ]


def rank_bars(owners, metrics, title, path):
    K = int(owners["grid"].max()) + 1
    x = np.arange(K)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for i, (key, hatch, label) in enumerate(
        [("grid", "", "grid"), ("mesh_0", "///", "G0"), ("mesh_1", "xx", "G1")]
    ):
        counts = np.bincount(owners[key], minlength=K)
        bars = axes[0].bar(
            x + (i - 1) * 0.25,
            counts,
            width=0.24,
            color=RANK_COLORS[:K],
            hatch=hatch,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[0].bar_label(bars, fontsize=7, padding=2, rotation=90)
    axes[0].legend(
        handles=[
            Patch(facecolor="#cccccc", hatch=h, label=n)
            for n, h in [("grid", ""), ("G0", "///"), ("G1", "xx")]
        ],
        fontsize=8,
    )
    axes[0].set(
        title="Owned nodes",
        ylabel="Node count",
        ylim=(0, max(np.bincount(owners["grid"])) * 1.3),
    )
    relations = list(dict.fromkeys(r["relation"] for r in metrics))
    bottom = np.zeros(K)
    for i, relation in enumerate(relations):
        counts = np.array(
            [
                next(
                    r["halo_nodes_received"]
                    for r in metrics
                    if r["rank"] == rank and r["relation"] == relation
                )
                for rank in range(K)
            ]
        )
        axes[1].bar(
            x,
            counts,
            bottom=bottom,
            color=plt.get_cmap("tab10")(i),
            label=relation,
        )
        bottom += counts
    for rank, total in enumerate(bottom):
        axes[1].text(
            rank, total, f"{int(total):,}", ha="center", va="bottom", fontsize=8
        )
    axes[1].set(
        title="Received halo nodes, stacked by relation",
        ylabel="Sum over one exchange per relation",
        ylim=(0, max(1, bottom.max()) * 1.3),
    )
    axes[1].legend(fontsize=7, ncol=2)
    for ax in axes:
        ax.set_xticks(x, [f"rank {r}" for r in x])
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle(title)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weighted", type=Path, default=DEFAULT)
    parser.add_argument(
        "--unweighted", type=Path, default=DEFAULT.parent / "1deg-unweighted"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results/partition-figures",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    info = graph_data()
    _, sizes, relations, store, fingerprint = info
    grid = store.get_xy("state", True)[store.get_mask(True, True, False)]
    coordinates = {"grid": np.asarray(grid, dtype=np.float64)}
    features = torch.load(
        store.root_path / "graphs" / GRAPH / "mesh_features.pt",
        weights_only=True,
    )
    for i, f in enumerate(features):
        slon, clon, slat, clat = f[:, :4].numpy().T
        assert np.allclose(slon**2 + clon**2, 1, atol=1e-5)
        assert np.allclose(slat**2 + clat**2, 1, atol=1e-5)
        coordinates[f"mesh_{i}"] = np.column_stack(
            (
                np.rad2deg(np.arctan2(slon, clon)),
                np.rad2deg(np.arctan2(slat, clat)),
            )
        )
    assert {name: len(xy) for name, xy in coordinates.items()} == sizes
    np.savez_compressed(args.output / "coordinates.npz", **coordinates)
    all_metrics, audits, cases = [], [], {}
    for variant, root, methods in [
        ("weighted", args.weighted, METHODS),
        (
            "unweighted",
            args.unweighted,
            tuple(m for m in METHODS if m != "chunks"),
        ),
    ]:
        for method in methods:
            for K in (2, 4):
                owners, _ = load_owners(root, method, K, info)
                owner_arrays = {
                    name: owner.numpy().astype(np.uint8)
                    for name, owner in owners.items()
                }
                cases[(variant, method, K)] = owner_arrays
                target = args.output / "owners" / variant
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    target / f"{method}-k{K}.npz",
                    K=np.int16(K),
                    partitioner=np.array(method),
                    graph_fingerprint=np.array(fingerprint),
                    **owner_arrays,
                )
                for name, owner in owner_arrays.items():
                    counts = np.bincount(owner, minlength=K)
                    assert (
                        len(owner) == sizes[name]
                        and counts.min() > 0
                        and owner.max() < K
                    )
                    imbalance = counts.max() / counts.mean()
                    audits.append(
                        dict(
                            assignment=variant,
                            partitioner=method,
                            K=K,
                            node_set=name,
                            owned_counts="/".join(map(str, counts)),
                            imbalance=float(imbalance),
                            status=(
                                "WARN"
                                if imbalance
                                > (1.05 if name == "grid" else 1.25)
                                else "OK"
                            ),
                        )
                    )
                metrics = []
                for relation, (src, dst, edge) in relations.items():
                    offsets, indices, _ = edge_index_to_csc(
                        edge, len(owners[dst])
                    )
                    for rank in range(K):
                        part = partition_graph_with_id_mapping(
                            offsets,
                            indices,
                            owners[src],
                            owners[dst],
                            K,
                            rank,
                            torch.device("cpu"),
                        )
                        metrics.append(
                            dict(
                                assignment=variant,
                                partitioner=method,
                                K=K,
                                rank=rank,
                                relation=relation,
                                hidden_dim=64,
                                batch_size=2,
                                **partition_metrics(part),
                                graph_fingerprint=fingerprint,
                            )
                        )
                all_metrics += metrics
                title = f"{variant} | {NAMES[method]} | K={K}"
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                fig.subplots_adjust(
                    left=0.01, right=0.99, bottom=0.10, top=0.86, wspace=0.02
                )
                for ax, center in zip(axes, (0, 180)):
                    globe(ax, coordinates, owner_arrays, center)
                    ax.set_title(f"Central longitude {center}°", fontsize=10)
                fig.suptitle(title)
                fig.legend(
                    handles=legends(K),
                    loc="lower center",
                    ncol=K + 3,
                    fontsize=8,
                )
                save(
                    fig,
                    args.output / "figures" / variant / f"{method}-k{K}-globe",
                )
                rank_bars(
                    owner_arrays,
                    metrics,
                    title,
                    args.output
                    / "figures"
                    / variant
                    / f"{method}-k{K}-rank-counts",
                )
                print(
                    f"PASS {title}: owners exported, both hemispheres and counts saved",
                    flush=True,
                )
    rows = [
        (v, m)
        for v, ms in [
            ("weighted", METHODS),
            ("unweighted", tuple(m for m in METHODS if m != "chunks")),
        ]
        for m in ms
    ]
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 22))
    fig.subplots_adjust(
        left=0.14, right=0.99, bottom=0.04, top=0.96, wspace=0.01, hspace=0.06
    )
    for i, (variant, method) in enumerate(rows):
        for j, (K, center) in enumerate([(2, 0), (2, 180), (4, 0), (4, 180)]):
            globe(
                axes[i, j],
                coordinates,
                cases[(variant, method, K)],
                center,
                small=True,
            )
            if i == 0:
                axes[i, j].set_title(f"K={K}, view {center}°", fontsize=10)
        axes[i, 0].text(
            -0.05,
            0.5,
            f"{variant}\n{NAMES[method]}",
            transform=axes[i, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
    fig.suptitle(
        "Actual ownership: weighted and unweighted partitions", fontsize=15
    )
    fig.legend(handles=legends(), loc="lower center", ncol=7, fontsize=10)
    save(fig, args.output / "overview")
    write_csv(args.output / "partition_metrics.csv", all_metrics)
    write_csv(args.output / "audit.csv", audits)
    result = dict(
        status="passed",
        graph_fingerprint=fingerprint,
        node_counts=sizes,
        partition_cases=len(cases),
        rank_files=42,
        owner_order="Loaded graph order, verified by load_owners (fingerprint, IDs, coordinates and relation edge IDs)",
        imbalance_warnings=[row for row in audits if row["status"] == "WARN"],
        audits=audits,
    )
    (args.output / "validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.output / "README.md").write_text("""# Partition figures and owners

All figures use the actual rank files. Weighted: chunks, RCB, METIS, Mt-KaHyPar.
Unweighted: RCB, METIS, Mt-KaHyPar. Chunks is included only once, as requested.
There are 14 partition cases (42 rank files), each with two-hemisphere globe views
and per-rank owned/halo bar charts, plus one overview. PNGs are 200 dpi; PDFs
rasterize node markers to keep file sizes small. Rank colors: 0 blue, 1 orange,
2 green, 3 purple. Grid markers are smallest; G0/G1 successively larger.

NPZ arrays grid/mesh_0/mesh_1 map graph-order node IDs to rank. coordinates.npz
has matching longitude/latitude arrays in degrees. No observation/sample arrays
are exported. Graph fingerprint validates graph and mesh node order; saved grid
coordinates and relation edge IDs validate their mapping against the loaded graph.

Halo bars sum received remote nodes over one exchange of each relation; a node
used in two relations counts twice. These are not unique global ghost nodes or
per-training-step traffic. partition_metrics.csv gives every relation separately,
with byte predictions for H64/FP32/batch2. audit.csv gives owned counts and max/mean
for every node set. Imbalance warnings for unweighted assignments are expected;
all structural ownership/graph-order checks passed.
""")
    print(
        f"Complete: {len(cases)} cases, {len(all_metrics)} relation/rank rows; output {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

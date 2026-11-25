"""Rebuild Package E owner maps, audit them, and optionally write rank data."""

import argparse
import csv
import ctypes
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from .partitions import CONFIG, DEFAULT, GRAPH, METHODS, graph_data


def csr(edges, n):
    edges = torch.cat((edges, edges.flip(0)), 1)
    keys = torch.unique(edges[0] * n + edges[1], sorted=True)
    keys = keys[keys // n != keys % n]
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            torch.bincount(keys // n, minlength=n).cumsum(0),
        )
    )
    return offsets.numpy(), (keys % n).numpy()


def metis_two_constraints(offsets, neighbors, weights, K):
    # Call the same METIS binary shipped by PyMetis: its Python API fixes ncon=1.
    # Null tpwgts/ubvec use equal targets and UFACTOR=30 for both constraints.
    import pymetis
    import pymetis._internal as internal

    dtype = np.dtype(pymetis.zero_copy_dtype())
    integer = ctypes.c_int64 if dtype.itemsize == 8 else ctypes.c_int32
    pointer = ctypes.POINTER(integer)
    library = ctypes.CDLL(internal.__file__)
    default = library.METIS_SetDefaultOptions
    default.argtypes = [pointer]
    default.restype = ctypes.c_int
    partition = library.METIS_PartGraphKway
    partition.argtypes = [pointer] * 8 + [ctypes.c_void_p] * 2 + [pointer] * 3
    partition.restype = ctypes.c_int

    def array(value):
        return np.ascontiguousarray(value, dtype=dtype)

    def ptr(value):
        return value.ctypes.data_as(pointer)

    n = array([len(weights)])
    constraints = array([2])
    parts = array([K])
    objective = array([0])
    offsets, neighbors = array(offsets), array(neighbors)
    vertex_weights = array(
        np.column_stack((np.ones(len(weights), dtype=dtype), weights)).reshape(
            -1
        )
    )
    options = np.empty(40, dtype=dtype)
    assert default(ptr(options)) == 1
    options[internal.options_indices.SEED] = 42
    options[internal.options_indices.UFACTOR] = 30
    options[internal.options_indices.OBJTYPE] = int(pymetis.ObjType.CUT)
    result = np.empty(len(weights), dtype=dtype)
    status = partition(
        ptr(n),
        ptr(constraints),
        ptr(offsets),
        ptr(neighbors),
        ptr(vertex_weights),
        None,
        None,
        ptr(parts),
        None,
        None,
        ptr(options),
        ptr(objective),
        ptr(result),
    )
    if status != 1:
        raise RuntimeError(f"METIS ncon=2 returned {status}")
    return torch.from_numpy(result.astype(np.int64))


def rcb(points, weights, K):
    owners = torch.empty(len(points), dtype=torch.long)

    def split(ids, start, ranks):
        if ranks == 1:
            owners[ids] = start
            return
        axis = (points[ids].amax(0) - points[ids].amin(0)).argmax()
        ids = ids[torch.argsort(points[ids, axis], stable=True)]
        cumulative = weights[ids].cumsum(0).double()
        left = ranks // 2
        legal = torch.arange(left, len(ids) - (ranks - left) + 1)
        cut = int(
            legal[
                (cumulative[legal - 1] - cumulative[-1] * left / ranks)
                .abs()
                .argmin()
            ]
        )
        split(ids[:cut], start, left)
        split(ids[cut:], start + left, ranks - left)

    split(torch.arange(len(points)), 0, K)
    return owners


def ratios(owners, K):
    return {
        key: float(torch.bincount(o, minlength=K).max()) / (len(o) / K)
        for key, o in owners.items()
    }


def acceptable(values):
    return values["grid"] <= 1.05 and all(
        v <= 1.25 for k, v in values.items() if k != "grid"
    )


def main():
    import pymetis
    import mtkahypar

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT.parent)
    parser.add_argument("--write-ranks", action="store_true")
    parser.add_argument(
        "--variant", choices=("weighted", "unweighted"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.variant is None:
        # The published maps were prepared in separate processes. Mt-KaHyPar's
        # runtime keeps state across calls even when set_seed(42) is repeated.
        for variant in ("unweighted", "weighted"):
            command = [
                sys.executable,
                "-m",
                "dphpc.pnemo.reweight",
                "--config-path",
                str(args.config_path),
                "--output",
                str(args.output),
                "--variant",
                variant,
            ]
            if args.write_ranks:
                command.append("--write-ranks")
            subprocess.run(command, check=True)
        return
    torch.set_num_threads(1)
    info = graph_data(args.config_path, GRAPH)
    _, sizes, relations, store, fingerprint = info
    features = torch.load(
        store.root_path / "graphs" / GRAPH / "mesh_features.pt",
        weights_only=True,
    )
    xyz = []
    for value in features:
        slon, clon, slat, clat = value[:, :4].unbind(1)
        xyz.append(torch.stack((clat * clon, clat * slon, slat), 1))
    xy = torch.as_tensor(
        store.get_xy("state", True)[store.get_mask(True, True, False)]
    )
    lon, lat = xy.deg2rad().unbind(1)
    grid_xyz = torch.stack(
        (lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()), 1
    )
    tree = cKDTree(xyz[0].double().numpy())
    attachments = {"grid": torch.from_numpy(tree.query(grid_xyz.numpy())[1])}
    attachments.update(
        {
            f"mesh_{i}": torch.from_numpy(tree.query(v.double().numpy())[1])
            for i, v in enumerate(xyz)
            if i > 0
        }
    )
    grid_weights = torch.bincount(
        attachments["grid"], minlength=sizes["mesh_0"]
    )
    assert grid_weights.min() > 0 and int(grid_weights.sum()) == sizes["grid"]
    n = len(grid_weights)
    edges = relations["m2m_0"][2]
    offsets, neighbors = csr(edges, n)
    runtime = mtkahypar.initialize(1)
    maps, rows = {}, []
    expected_root = Path(__file__).with_name("owners")

    def inherit(g0):
        return {
            "mesh_0": g0,
            **{key: g0[ids] for key, ids in attachments.items()},
        }

    for variant in (args.variant,):
        for method in METHODS:
            if variant == "unweighted" and method == "chunks":
                continue
            for K in (2, 4):
                weights = (
                    grid_weights
                    if variant == "weighted"
                    else torch.ones(n, dtype=torch.long)
                )
                if method == "chunks":
                    owners = {
                        key: torch.arange(size) * K // size
                        for key, size in sizes.items()
                    }
                else:
                    if method == "rcb":
                        g0 = rcb(xyz[0], weights, K)
                    elif method == "metis":
                        result = pymetis.part_graph(
                            K,
                            adjacency=pymetis.CSRAdjacency(
                                offsets.tolist(), neighbors.tolist()
                            ),
                            vweights=weights.tolist(),
                            recursive=False,
                            options=pymetis.Options(
                                seed=42, ufactor=30, objtype=pymetis.ObjType.CUT
                            ),
                        )
                        g0 = torch.tensor(result.vertex_part, dtype=torch.long)
                        if variant == "weighted" and not acceptable(
                            ratios(inherit(g0), K)
                        ):
                            g0 = metis_two_constraints(
                                offsets, neighbors, weights.numpy(), K
                            )
                    else:
                        context = runtime.context_from_preset(
                            mtkahypar.PresetType.DEFAULT
                        )
                        context.set_partitioning_parameters(
                            K, 0.03, mtkahypar.Objective.KM1
                        )
                        context.logging = False
                        mtkahypar.set_seed(42)
                        ids = torch.arange(n)
                        keys = torch.unique(
                            torch.cat((edges[0] * n + edges[1], ids * n + ids)),
                            sorted=True,
                        )
                        nets = [
                            v.tolist()
                            for v in torch.split(
                                keys % n,
                                torch.bincount(keys // n, minlength=n).tolist(),
                            )
                        ]
                        hypergraph = runtime.create_hypergraph(
                            context, n, n, nets, weights.tolist(), [1] * n
                        )
                        g0 = torch.tensor(
                            hypergraph.partition(context).get_partition(),
                            dtype=torch.long,
                        )
                    owners = inherit(g0)
                expected = np.load(
                    expected_root / variant / f"{method}-k{K}.npz"
                )
                assert expected["graph_fingerprint"].item() == fingerprint
                for key, size in sizes.items():
                    owner = owners[key]
                    assert owner.shape == (size,) and bool(
                        ((owner >= 0) & (owner < K)).all()
                    )
                    counts = torch.bincount(owner, minlength=K).tolist()
                    assert min(counts) > 0
                    actual = owner.numpy().astype(np.uint8)
                    reference = expected[key]
                    actual_hash = hashlib.sha256(actual.tobytes()).hexdigest()
                    expected_hash = hashlib.sha256(
                        reference.tobytes()
                    ).hexdigest()
                    assert (
                        actual_hash == expected_hash
                    ), f"{variant}/{method}/k{K}/{key}: rebuilt map differs from final job"
                    imbalance = max(counts) / (size / K)
                    limit = 1.05 if key == "grid" else 1.25
                    row = dict(
                        variant=variant,
                        partitioner=method,
                        K=K,
                        node_set=key,
                        counts="/".join(map(str, counts)),
                        max_mean=imbalance,
                        status="WARN" if imbalance > limit else "OK",
                        sha256=actual_hash,
                    )
                    rows.append(row)
                    print(
                        f"{variant:10} {method:16} K={K} {key:6} {row['counts']:26} {imbalance:.3f} {row['status']} HASH PASS",
                        flush=True,
                    )
                if variant == "weighted" and method != "chunks":
                    assert int(
                        torch.bincount(owners["grid"], minlength=K).max()
                    ) <= math.ceil(1.03 * sizes["grid"] / K)
                    assert acceptable(ratios(owners, K))
                maps[variant, method, K] = owners
                output = args.output / "owners" / variant / f"{method}-k{K}.npz"
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output,
                    **{
                        key: value.numpy().astype(np.uint8)
                        for key, value in owners.items()
                    },
                    K=K,
                    partitioner=method,
                    graph_fingerprint=fingerprint,
                )
    with (args.output / f"ownership-audit-{args.variant}.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    if not args.write_ranks:
        return

    # Same raw arrays, metadata and spatial slicing as the original rank writer.
    from neural_lam.config import load_config_and_datastores

    _, ocean, boundary, atmosphere, *_ = load_config_and_datastores(
        str(args.config_path)
    )
    assert boundary is None and not ocean.is_forecast
    surface = ocean.get_mask(surface=True, stacked=True, invert=False)
    sea = np.flatnonzero(surface)
    np.testing.assert_array_equal(
        ocean._ds.time.values, atmosphere._ds.time.values
    )
    np.testing.assert_array_equal(
        ocean.get_xy("state", True), atmosphere.get_xy("forcing", True)
    )
    raw, normalization, features = {}, {}, {}
    for name, datastore, category in [
        ("state", ocean, "state"),
        ("forcing", ocean, "forcing"),
        ("atmosphere", atmosphere, "forcing"),
    ]:
        array = datastore._ds[category].transpose(
            "time", "grid_index", f"{category}_feature"
        )
        raw[name] = torch.from_numpy(
            np.ascontiguousarray(array.isel(grid_index=sea).values)
        )
        normalization[name] = {
            key: torch.as_tensor(value.values.copy())
            for key, value in datastore.get_standardization_dataarray(
                category
            ).data_vars.items()
        }
        features[name] = datastore.get_vars_names(category)
    land = torch.from_numpy(
        ocean.get_mask(surface=False, stacked=True, invert=True)[surface].copy()
    )
    template = dict(
        imbalance_tolerance=0.03,
        assignment="nearest G0 on unit sphere",
        graph=GRAPH,
        graph_fingerprint=fingerprint,
        time_ns=torch.from_numpy(
            ocean._ds.time.values.astype("datetime64[ns]").astype("int64")
        ),
        splits={
            str(name): [
                str(v) for v in ocean._ds.splits.sel(split_name=name).values
            ]
            for name in ocean.available_splits
        },
        normalization=normalization,
        features=features,
        training_config=args.config_path.read_text(),
    )
    for (variant, method, K), owners in maps.items():
        directory = (
            args.output
            / ("1deg" if variant == "weighted" else "1deg-unweighted")
            / method
            / f"k{K}"
        )
        if directory.exists():
            raise FileExistsError(directory)
        directory.mkdir(parents=True)
        for rank in range(K):
            nodes = {
                key: torch.where(value == rank)[0]
                for key, value in owners.items()
            }
            grid = nodes["grid"]
            local = {}
            for key, (src, dst, edge) in relations.items():
                edge_ids = torch.where(owners[dst][edge[1]] == rank)[0]
                send = edge[0, edge_ids]
                halo = torch.unique(
                    send[owners[src][send] != rank], sorted=True
                )
                local[key] = dict(
                    edge_ids=edge_ids,
                    halo_nodes=halo,
                    halo_owners=owners[src][halo],
                )
            shard = dict(
                template,
                partitioner=method,
                K=K,
                rank=rank,
                node_ids=nodes,
                relations=local,
                balance=variant,
                data={
                    key: value.index_select(1, grid)
                    for key, value in raw.items()
                },
                land_mask=land[grid],
                grid_lon_lat_deg=xy[grid],
            )
            path = directory / f"rank{rank}.pt"
            torch.save(shard, path)
            saved = torch.load(path, weights_only=True, mmap=True)
            for key, value in raw.items():
                torch.testing.assert_close(
                    saved["data"][key],
                    value.index_select(1, grid),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
    from .partitions import audit, load_owners

    if args.variant == "weighted":
        audit(args.output / "1deg", info)
    else:
        for method in METHODS[1:]:
            for K in (2, 4):
                load_owners(args.output / "1deg-unweighted", method, K, info)
    print(f"PASS: {args.variant} regenerated rank files validated", flush=True)


if __name__ == "__main__":
    main()

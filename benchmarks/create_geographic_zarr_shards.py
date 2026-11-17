"""Create rank-local Zarr shards for geographic model partitions.

The source MDP stores use one spatial chunk containing the complete global
grid.  Selecting a rank's nodes only after opening those stores therefore
does not reduce physical I/O.  This command materializes the valid ocean
surface nodes owned by each rank into independent Zarr stores.

The output is deliberately a small rank-local interchange format rather than
an MDP datastore: geographic RCB partitions are irregular subsets of the
global longitude/latitude grid, whereas MDP datastores assume a complete
Cartesian grid.  ``global_grid_index`` and ``global_surface_index`` preserve
the exact mapping back to the source data and the Graph-EFM grid nodes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from neural_lam.geographic_partitioning import (
    SUPPORTED_WORLD_SIZES,
    geographic_rcb_owners,
    partition_indices,
)


FORMAT_VERSION = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create validated two- and four-rank Zarr shards."
    )
    parser.add_argument(
        "--ocean-store",
        type=Path,
        default=Path("data/global_ocean_1_mdp_config.zarr"),
    )
    parser.add_argument(
        "--atmosphere-store",
        type=Path,
        default=Path("data/global_era5_1_mdp_config.zarr"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/partitioned/global_1"),
    )
    parser.add_argument(
        "--world-sizes",
        type=int,
        nargs="+",
        default=list(SUPPORTED_WORLD_SIZES),
    )
    parser.add_argument("--longitude-seam", type=float, default=120.0)
    parser.add_argument(
        "--skip-value-validation",
        action="store_true",
        help="Only validate ownership and metadata, not every array value.",
    )
    return parser.parse_args()


def _open_source(path: Path) -> xr.Dataset:
    if not path.is_dir():
        raise FileNotFoundError(f"Zarr store does not exist: {path}")
    return xr.open_zarr(path, consolidated=True)


def _source_layout(
    ocean: xr.Dataset,
    atmosphere: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required_ocean = {"state", "forcing", "static", "mask"}
    missing = required_ocean.difference(ocean.variables)
    if missing:
        raise ValueError(f"ocean store is missing variables: {sorted(missing)}")
    if "forcing" not in atmosphere:
        raise ValueError("atmosphere store is missing forcing")

    ocean_size = ocean.sizes.get("grid_index")
    atmosphere_size = atmosphere.sizes.get("grid_index")
    if ocean_size is None or ocean_size != atmosphere_size:
        raise ValueError("ocean and atmosphere grid sizes do not match")

    ocean_longitude = ocean.longitude.values
    ocean_latitude = ocean.latitude.values
    np.testing.assert_array_equal(
        atmosphere.longitude.values,
        ocean_longitude,
        err_msg="ocean and atmosphere longitudes do not match",
    )
    np.testing.assert_array_equal(
        atmosphere.latitude.values,
        ocean_latitude,
        err_msg="ocean and atmosphere latitudes do not match",
    )

    surface_mask = ocean["mask"].isel(mask_feature=0).values.astype(bool)
    if surface_mask.shape != (ocean_size,):
        raise ValueError("surface mask does not match the global grid")
    surface_grid_indices = np.flatnonzero(surface_mask).astype(np.int64)
    surface_lon_lat = np.column_stack(
        (
            ocean_longitude[surface_grid_indices],
            ocean_latitude[surface_grid_indices],
        )
    )
    return surface_mask, surface_grid_indices, surface_lon_lat


def _chunk_rank_local(dataset: xr.Dataset) -> xr.Dataset:
    chunks = {}
    for dimension, size in dataset.sizes.items():
        if dimension in {"time", "init_time", "lead_time"}:
            chunks[dimension] = 1
        else:
            chunks[dimension] = size
    return dataset.chunk(chunks)


def _write_rank_store(
    source: xr.Dataset,
    target: Path,
    global_grid_indices: np.ndarray,
    global_surface_indices: np.ndarray,
    metadata: dict,
) -> None:
    local_size = len(global_grid_indices)
    shard = source.isel(grid_index=global_grid_indices).assign_coords(
        grid_index=np.arange(local_size, dtype=np.int64),
        global_grid_index=("grid_index", global_grid_indices),
        global_surface_index=("grid_index", global_surface_indices),
    )
    shard.coords["grid_index"].attrs.update(
        long_name="rank-local contiguous grid index"
    )
    shard.coords["global_grid_index"].attrs.update(
        long_name="grid_index in the original full Cartesian source store"
    )
    shard.coords["global_surface_index"].attrs.update(
        long_name="node index in the global valid-ocean Graph-EFM grid"
    )
    shard.attrs.update(metadata)
    shard = _chunk_rank_local(shard)
    shard.to_zarr(
        target,
        mode="w",
        consolidated=True,
        zarr_format=2,
    )


def _assert_values_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    if np.issubdtype(actual.dtype, np.inexact):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    else:
        np.testing.assert_array_equal(actual, expected)


def _validate_store(
    source: xr.Dataset,
    target: Path,
    global_grid_indices: np.ndarray,
    global_surface_indices: np.ndarray,
    validate_values: bool,
) -> None:
    shard = xr.open_zarr(target, consolidated=True)
    local_size = len(global_grid_indices)
    if shard.sizes["grid_index"] != local_size:
        raise AssertionError(f"wrong grid size in {target}")
    np.testing.assert_array_equal(
        shard.global_grid_index.values,
        global_grid_indices,
    )
    np.testing.assert_array_equal(
        shard.global_surface_index.values,
        global_surface_indices,
    )

    if not validate_values:
        return
    for name, variable in source.variables.items():
        if name == "grid_index":
            continue
        expected = variable
        if "grid_index" in variable.dims:
            expected = expected.isel(grid_index=global_grid_indices)
        _assert_values_equal(shard[name].values, expected.values)


def _directory_size(path: Path) -> int:
    return sum(
        item.stat().st_size for item in path.rglob("*") if item.is_file()
    )


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _create_world_size(
    ocean: xr.Dataset,
    atmosphere: xr.Dataset,
    ocean_path: Path,
    atmosphere_path: Path,
    output_root: Path,
    world_size: int,
    longitude_seam: float,
    surface_grid_indices: np.ndarray,
    surface_lon_lat: np.ndarray,
    validate_values: bool,
) -> dict:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"world size must be one of {SUPPORTED_WORLD_SIZES}, got "
            f"{world_size}"
        )

    final_dir = output_root / f"geographic_rcb_ws{world_size}"
    if final_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing partition directory: {final_dir}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_dir.name}.tmp-",
            dir=output_root,
        )
    )

    try:
        owners = geographic_rcb_owners(
            torch.as_tensor(surface_lon_lat),
            world_size=world_size,
            longitude_seam=longitude_seam,
        )
        local_surface_indices = partition_indices(owners, world_size)
        np.save(staging_dir / "surface_owners.npy", owners.numpy())

        ranks = []
        all_surface_indices = []
        all_grid_indices = []
        for rank, rank_surface_tensor in enumerate(local_surface_indices):
            rank_surface = rank_surface_tensor.numpy().astype(np.int64)
            rank_grid = surface_grid_indices[rank_surface]
            rank_dir = staging_dir / f"rank_{rank}"
            rank_dir.mkdir()
            common_metadata = {
                "partition_format": "neural_lam_surface_shard",
                "partition_format_version": FORMAT_VERSION,
                "partition_strategy": "geographic_rcb",
                "partition_world_size": world_size,
                "partition_rank": rank,
                "partition_longitude_seam_degrees_east": longitude_seam,
                "global_cartesian_grid_size": int(ocean.sizes["grid_index"]),
                "global_surface_grid_size": int(len(surface_grid_indices)),
                "local_surface_grid_size": int(len(rank_grid)),
            }
            _write_rank_store(
                ocean,
                rank_dir / "ocean.zarr",
                rank_grid,
                rank_surface,
                {**common_metadata, "partition_source": ocean_path.name},
            )
            _write_rank_store(
                atmosphere,
                rank_dir / "atmosphere.zarr",
                rank_grid,
                rank_surface,
                {
                    **common_metadata,
                    "partition_source": atmosphere_path.name,
                },
            )
            _validate_store(
                ocean,
                rank_dir / "ocean.zarr",
                rank_grid,
                rank_surface,
                validate_values,
            )
            _validate_store(
                atmosphere,
                rank_dir / "atmosphere.zarr",
                rank_grid,
                rank_surface,
                validate_values,
            )
            all_surface_indices.append(rank_surface)
            all_grid_indices.append(rank_grid)
            ranks.append(
                {
                    "rank": rank,
                    "local_surface_grid_size": int(len(rank_grid)),
                    "ocean_store": f"rank_{rank}/ocean.zarr",
                    "atmosphere_store": f"rank_{rank}/atmosphere.zarr",
                    "minimum_latitude": float(
                        surface_lon_lat[rank_surface, 1].min()
                    ),
                    "maximum_latitude": float(
                        surface_lon_lat[rank_surface, 1].max()
                    ),
                }
            )

        np.testing.assert_array_equal(
            np.sort(np.concatenate(all_surface_indices)),
            np.arange(len(surface_grid_indices), dtype=np.int64),
        )
        np.testing.assert_array_equal(
            np.sort(np.concatenate(all_grid_indices)),
            surface_grid_indices,
        )
        if len(np.unique(np.concatenate(all_grid_indices))) != len(
            surface_grid_indices
        ):
            raise AssertionError("a global grid point has multiple owners")

        manifest = {
            "format": "neural_lam_surface_partition",
            "format_version": FORMAT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "strategy": "geographic_rcb",
            "world_size": world_size,
            "longitude_seam_degrees_east": longitude_seam,
            "source_ocean_store": ocean_path.name,
            "source_atmosphere_store": atmosphere_path.name,
            "global_cartesian_grid_size": int(ocean.sizes["grid_index"]),
            "global_surface_grid_size": int(len(surface_grid_indices)),
            "rank_counts": [rank["local_surface_grid_size"] for rank in ranks],
            "ranks": ranks,
            "validation": {
                "ownership_complete": True,
                "ownership_disjoint": True,
                "coordinates_match_sources": True,
                "all_array_values_match_sources": validate_values,
            },
        }
        _write_json(staging_dir / "manifest.json", manifest)
        staging_dir.replace(final_dir)
        manifest["size_bytes"] = _directory_size(final_dir)
        _write_json(final_dir / "manifest.json", manifest)
        return manifest
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def main() -> None:
    args = _parse_args()
    if len(set(args.world_sizes)) != len(args.world_sizes):
        raise ValueError("world sizes must be unique")

    ocean_path = args.ocean_store.resolve()
    atmosphere_path = args.atmosphere_store.resolve()
    output_root = args.output_dir.resolve()
    ocean = _open_source(ocean_path)
    atmosphere = _open_source(atmosphere_path)
    _, surface_grid_indices, surface_lon_lat = _source_layout(
        ocean,
        atmosphere,
    )

    print(
        "source: "
        f"{ocean.sizes['grid_index']:,} Cartesian points, "
        f"{len(surface_grid_indices):,} valid ocean points"
    )
    for world_size in args.world_sizes:
        manifest = _create_world_size(
            ocean=ocean,
            atmosphere=atmosphere,
            ocean_path=ocean_path,
            atmosphere_path=atmosphere_path,
            output_root=output_root,
            world_size=world_size,
            longitude_seam=args.longitude_seam,
            surface_grid_indices=surface_grid_indices,
            surface_lon_lat=surface_lon_lat,
            validate_values=not args.skip_value_validation,
        )
        size_mib = manifest["size_bytes"] / 1024**2
        print(
            f"world_size={world_size}: counts={manifest['rank_counts']}, "
            f"validated={manifest['validation']}, size={size_mib:.1f} MiB"
        )


if __name__ == "__main__":
    main()

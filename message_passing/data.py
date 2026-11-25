"""Original WeatherDataset with explicit storage and ownership validation."""

# Standard library
import json
from pathlib import Path

# Third-party
import torch
import xarray as xr
from torch.utils.data import DataLoader, Subset

# First-party
from neural_lam.datastore.mdp import MDPDatastore

# Local
from .geographic import geographic_rcb_owners
from .partition import graph_fingerprint, model_graph, validate_owners
from .workload import make_dataset, make_model_args


def ownership(model, world_size, owner_file=None):
    counts, _ = model_graph(model)
    if owner_file:
        saved = torch.load(owner_file, weights_only=True, map_location="cpu")
        if (
            saved["graph_sha256"] != graph_fingerprint(model)
            or saved["world_size"] != world_size
        ):
            raise ValueError(
                "Ownership file belongs to a different graph or world size"
            )
        owners = saved["owners"]
    else:
        store = model._datastore
        grid = torch.as_tensor(
            store.get_xy("state", stacked=True)[model.surface_mask]
        )
        # Mesh raw features are [sin(lon), cos(lon), sin(lat), cos(lat), ...].
        raw = torch.load(
            store.root_path / "graphs" / model.args.graph / "mesh_features.pt",
            weights_only=True,
        )
        coordinates = {"grid": grid}
        for i, features in enumerate(raw):
            slon, clon, slat, clat = features[:, :4].unbind(1)
            coordinates[f"mesh_{i}"] = torch.rad2deg(
                torch.stack(
                    (torch.atan2(slon, clon), torch.atan2(slat, clat)), 1
                )
            )
        owners = {
            name: geographic_rcb_owners(x, world_size, longitude_seam=120.0)
            for name, x in coordinates.items()
        }
    if set(owners) != set(counts):
        raise ValueError("Ownership node sets differ from the model graph")
    for name, count in counts.items():
        validate_owners(owners[name], count, world_size, name)
    return owners


def _shard_store(config_path, store_path):
    if not store_path.is_dir():
        raise FileNotFoundError(store_path)
    store = MDPDatastore(config_path)
    store._ds = xr.open_zarr(store_path, consolidated=True)
    store._available_splits = list(store._ds.splits.split_name.values)
    return store


def rank_dataset(args, config, model, owners, rank, world_size):
    root = Path(args.shard_root) / f"geographic_rcb_ws{world_size}"
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["world_size"] != world_size or manifest[
        "global_surface_grid_size"
    ] != len(owners["grid"]):
        raise ValueError("Shard manifest differs from the model")
    if config.datastore_boundary is not None:
        raise NotImplementedError("Boundary datastores are unsupported")
    base = Path(args.config_path).parent
    ocean = _shard_store(
        base / config.datastore.config_path, root / f"rank_{rank}/ocean.zarr"
    )
    atmosphere = None
    if config.datastore_atmosphere:
        atmosphere = _shard_store(
            base / config.datastore_atmosphere.config_path,
            root / f"rank_{rank}/atmosphere.zarr",
        )
    expected_ids = torch.where(owners["grid"] == rank)[0]
    expected_grid = torch.where(torch.as_tensor(model.surface_mask))[0][
        expected_ids
    ]
    for store in (ocean, atmosphere):
        if store is not None:
            for name, expected in (
                ("global_surface_index", expected_ids),
                ("global_grid_index", expected_grid),
            ):
                if not torch.equal(
                    torch.as_tensor(store._ds[name].values), expected
                ):
                    raise ValueError(
                        f"Shard {name} differs from ownership; "
                        "regenerate data for this partition"
                    )
    return make_dataset(
        args,
        make_model_args(args),
        config,
        ocean,
        None,
        atmosphere,
        statistics_datastore=ocean,
        statistics_datastore_atmosphere=atmosphere,
    )


def data_loader(dataset, args, *, finite_cache=False):
    options = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        generator=torch.Generator().manual_seed(args.seed),
    )
    if args.workers:
        options.update(
            multiprocessing_context="spawn",
            prefetch_factor=args.prefetch,
            persistent_workers=not finite_cache,
        )
    if finite_cache:
        count = args.cached_batches * args.batch_size
        if len(dataset) < count:
            raise ValueError(
                f"Need {count} distinct samples, got {len(dataset)}"
            )
        dataset = Subset(dataset, range(count))
    return DataLoader(dataset, **options)


def local_cpu_batch(batch, nodes, already_local):
    return tuple(
        (
            t.index_select(-2, nodes.cpu())
            if i in (0, 1, 2, 4) and not already_local
            else t
        )
        for i, t in enumerate(batch)
    )

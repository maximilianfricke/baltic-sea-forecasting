"""Original data input and externally supplied graph ownership."""

# Standard library
from pathlib import Path

# Third-party
import torch
from torch.utils.data import DataLoader

# Local
from .geographic import geographic_rcb_owners
from .partition import graph_fingerprint, model_graph, validate_owners


def ownership(model, world_size, owner_file=None):
    counts, _ = model_graph(model)
    if owner_file:
        saved = torch.load(owner_file, weights_only=True, map_location="cpu")
        if (
            saved["graph_sha256"] != graph_fingerprint(model)
            or saved["world_size"] != world_size
        ):
            raise ValueError(
                "Ownership belongs to a different graph or GPU count"
            )
        owners = saved["owners"]
    else:
        store = model._datastore
        coordinates = {
            "grid": torch.as_tensor(
                store.get_xy("state", stacked=True)[model.surface_mask]
            )
        }
        raw = torch.load(
            Path(store.root_path)
            / "graphs"
            / model.args.graph
            / "mesh_features.pt",
            weights_only=True,
        )
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
        raise ValueError("Ownership node sets differ from the original graph")
    for name, count in counts.items():
        validate_owners(owners[name], count, world_size, name)
    return owners


def data_loader(dataset, batch_size, workers, seed):
    """Live original samples; all spatial ranks must see the same batch IDs.

    No DistributedSampler: ranks split space, not examples. Prefetch is the
    DataLoader default (2 with workers), as in original WeatherDataModule.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        multiprocessing_context="spawn" if workers else None,
        generator=torch.Generator().manual_seed(seed),
    )


def local_batch(batch, grid_ids):
    """Select owned ocean nodes after the original WeatherDataset processing."""
    return tuple(
        t.index_select(-2, grid_ids.cpu()) if i in (0, 1, 2, 4) else t
        for i, t in enumerate(batch)
    )

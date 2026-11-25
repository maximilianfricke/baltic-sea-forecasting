"""Use the original WeatherDataset on owned raw fields and saved global statistics."""

import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from neural_lam.datastore.base import BaseDatastore

from .workload import make_dataset, make_model_args


class RankStore:
    """The datastore methods WeatherDataset needs, backed only by one rank file."""

    is_forecast = False
    is_ensemble = False

    def __init__(self, shard, atmosphere=False):
        self.fields = (
            {"forcing": "atmosphere"}
            if atmosphere
            else {"state": "state", "forcing": "forcing"}
        )
        self.data = {
            category: shard["data"][key].numpy()
            for category, key in self.fields.items()
        }
        self.stats = {
            category: shard["normalization"][key]
            for category, key in self.fields.items()
        }
        self.features = {
            category: shard["features"][key]
            for category, key in self.fields.items()
        }
        self.time = shard["time_ns"].numpy().view("datetime64[ns]")
        self.splits = shard["splits"]
        self.grid = shard["node_ids"]["grid"].numpy()
        self.land = shard["land_mask"].numpy()

    def get_dataarray(self, category, split):
        feature = f"{category}_feature"
        array = xr.DataArray(
            self.data[category],
            dims=("time", "grid_index", feature),
            coords={
                "time": self.time,
                "grid_index": self.grid,
                feature: self.features[category],
            },
        )
        start, end = self.splits[split]
        return array.sel(time=slice(start, end))

    def get_vars_names(self, category):
        return self.features[category]

    def expected_dim_order(self, category):
        return BaseDatastore.expected_dim_order(self, category)

    def get_standardization_dataarray(self, category):
        feature = f"{category}_feature"
        return xr.Dataset(
            {
                name: (feature, value.numpy())
                for name, value in self.stats[category].items()
            },
            coords={feature: self.features[category]},
        )

    def get_mask(self, surface, stacked, invert):
        mask = np.ones(len(self.grid), dtype=bool) if surface else ~self.land
        return ~mask if invert else mask


class RankDataset(Dataset):
    def __init__(self, args, config, shard, full_grid_size):
        self.weather = make_dataset(
            args,
            make_model_args(args),
            config,
            RankStore(shard),
            None,
            RankStore(shard, atmosphere=True),
        )
        self.full_grid_size = full_grid_size

    def __len__(self):
        return len(self.weather)

    def __getitem__(self, index):
        sample = self.weather[index]
        # Preserve the original absent-boundary tensor's shape (zero elements).
        boundary = sample[3].new_empty(
            (self.weather.ar_steps, self.full_grid_size, 0)
        )
        return (*sample[:3], boundary, *sample[4:])

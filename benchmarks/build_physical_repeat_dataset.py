"""Build a self-contained physical-repeat dataset for I/O benchmarks.

This deliberately repeats values while writing independent Zarr chunks and
monotonically increasing timestamps. It is an I/O proxy, not additional
scientific training data. The source directory is never modified.
"""

# Standard library
import argparse
import shutil
from pathlib import Path

# Third-party
import numpy as np
import xarray as xr


CONFIG_FILES = (
    "global_ocean_1.yaml",
    "global_ocean_1_mdp_config.yaml",
    "global_era5_1_mdp_config.yaml",
)
STORE_NAMES = (
    "global_ocean_1_mdp_config.zarr",
    "global_era5_1_mdp_config.zarr",
)


def _format_time(value):
    return np.datetime_as_string(value, unit="m")


def _split_length(dataset, name):
    split = dataset.splits.sel(split_name=name)
    start = np.datetime64(split.sel(split_part="start").load().item())
    end = np.datetime64(split.sel(split_part="end").load().item())
    times = dataset.time.values
    return int(np.count_nonzero((times >= start) & (times <= end)))


def _repeat_store(source, destination, repeats):
    dataset = xr.open_zarr(source, consolidated=True)
    try:
        original_times = dataset.time.values
        if original_times.size < 2:
            raise ValueError(f"{source} needs at least two timestamps")
        step = original_times[1] - original_times[0]
        span = original_times[-1] - original_times[0] + step
        time_variables = [
            name
            for name, variable in dataset.data_vars.items()
            if "time" in variable.dims
        ]
        if not time_variables:
            raise ValueError(f"{source} has no time-dependent variables")

        blocks = []
        for repeat_index in range(repeats):
            block = dataset[time_variables].assign_coords(
                time=dataset.time + repeat_index * span
            )
            blocks.append(block)
        dynamic = xr.concat(
            blocks,
            dim="time",
            data_vars="all",
            coords="minimal",
            compat="override",
        )
        static = dataset.drop_dims("time")
        repeated = xr.merge([static, dynamic], compat="override")

        validation_length = _split_length(dataset, "val")
        test_length = _split_length(dataset, "test")
        total_length = repeated.sizes["time"]
        training_length = total_length - validation_length - test_length
        if training_length < 1:
            raise ValueError("Repeated dataset leaves no training samples")
        times = repeated.time.values
        split_values = np.array(
            [
                [
                    _format_time(times[0]),
                    _format_time(times[training_length - 1]),
                ],
                [
                    _format_time(times[training_length]),
                    _format_time(
                        times[training_length + validation_length - 1]
                    ),
                ],
                [
                    _format_time(times[training_length + validation_length]),
                    _format_time(times[-1]),
                ],
            ]
        )
        repeated["splits"] = xr.DataArray(
            split_values,
            dims=("split_name", "split_part"),
            coords={
                "split_name": ["train", "val", "test"],
                "split_part": ["start", "end"],
            },
        )
        repeated.attrs.update(
            {
                "benchmark_data_kind": "physical_repeat_io_proxy",
                "benchmark_source_store": source.name,
                "benchmark_repeat_count": repeats,
                "benchmark_original_time_steps": original_times.size,
                "benchmark_effective_time_steps": total_length,
            }
        )
        repeated = repeated.chunk({"time": 1})
        repeated.to_zarr(
            destination,
            mode="w",
            consolidated=True,
            zarr_format=2,
        )
        return {
            "original": int(original_times.size),
            "effective": int(total_length),
            "train": int(training_length),
            "val": int(validation_length),
            "test": int(test_length),
        }
    finally:
        dataset.close()


def _verify_store(path, expected_steps, repeats):
    dataset = xr.open_zarr(path, consolidated=True)
    try:
        if dataset.sizes["time"] != expected_steps:
            raise ValueError(f"Unexpected time length in {path}")
        if dataset.attrs.get("benchmark_repeat_count") != repeats:
            raise ValueError(f"Missing repeat metadata in {path}")
        times = dataset.time.values
        if not np.all(np.diff(times) > np.timedelta64(0, "ns")):
            raise ValueError(
                f"Timestamps are not strictly increasing in {path}"
            )
        for category in ("state", "forcing"):
            if category not in dataset:
                continue
            first = dataset[category].isel(time=0).values
            repeated_first = dataset[category].isel(
                time=expected_steps // repeats
            ).values
            if not np.array_equal(first, repeated_first, equal_nan=True):
                raise ValueError(f"Repeated values differ for {category}")
    finally:
        dataset.close()


def build(source_root, destination_root, repeats):
    if repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if destination_root.exists():
        raise FileExistsError(f"Destination already exists: {destination_root}")
    building_root = destination_root.with_name(
        f"{destination_root.name}.building"
    )
    if building_root.exists():
        raise FileExistsError(
            f"Incomplete build directory already exists: {building_root}"
        )
    building_root.mkdir(parents=True)

    for name in CONFIG_FILES:
        shutil.copy2(source_root / name, building_root / name)
    shutil.copytree(source_root / "graphs", building_root / "graphs")

    summaries = {}
    for name in STORE_NAMES:
        summaries[name] = _repeat_store(
            source_root / name,
            building_root / name,
            repeats,
        )
        _verify_store(
            building_root / name,
            summaries[name]["effective"],
            repeats,
        )
    building_root.rename(destination_root)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=16)
    args = parser.parse_args()
    summaries = build(
        args.source_root.resolve(),
        args.destination_root.resolve(),
        args.repeats,
    )
    for name, summary in summaries.items():
        print(f"{name}: {summary}")
    print(f"Physical I/O proxy written to {args.destination_root.resolve()}")


if __name__ == "__main__":
    main()

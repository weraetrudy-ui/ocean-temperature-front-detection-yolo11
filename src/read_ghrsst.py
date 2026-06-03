from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


SUPPORTED_SUFFIXES = {".nc", ".nc4", ".cdf"}
SST_CANDIDATES = (
    "analysed_sst",
    "sst",
    "sea_surface_temperature",
    "sea_surface_temp",
    "temperature",
    "temp",
)


def list_nc_files(input_dir: Path) -> list[Path]:
    files = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {input_dir}")
    return files


def guess_name(ds_or_da: xr.Dataset | xr.DataArray, candidates: tuple[str, ...] | list[str]) -> str | None:
    names = list(ds_or_da.coords) + list(ds_or_da.dims)
    lower_map = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def find_sst_var(ds: xr.Dataset, lat_name: str | None = None, lon_name: str | None = None) -> str:
    lower_map = {name.lower(): name for name in ds.data_vars}
    for candidate in SST_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for name, data_array in ds.data_vars.items():
        dims = set(data_array.dims)
        if lat_name in dims and lon_name in dims:
            return name
        if data_array.ndim >= 2 and np.issubdtype(data_array.dtype, np.number):
            return name

    raise ValueError("No SST-like variable found")


def find_time_dim(data_array: xr.DataArray, lat_name: str, lon_name: str) -> str | None:
    for dim_name in data_array.dims:
        if dim_name in {lat_name, lon_name}:
            continue
        if dim_name.lower() in {"time", "times", "date", "day"}:
            return dim_name

    other_dims = [dim for dim in data_array.dims if dim not in {lat_name, lon_name}]
    return other_dims[0] if len(other_dims) == 1 else None


def to_celsius(values: np.ndarray) -> np.ndarray:
    sst = np.asarray(values, dtype=float)
    finite = sst[np.isfinite(sst)]
    if finite.size and float(np.nanmedian(finite)) > 100.0:
        return sst - 273.15
    return sst


def describe_file(file_path: Path) -> None:
    with xr.open_dataset(file_path) as ds:
        lat_name = guess_name(ds, ("lat", "latitude", "y"))
        lon_name = guess_name(ds, ("lon", "longitude", "x"))
        sst_var = find_sst_var(ds, lat_name, lon_name)
        da = ds[sst_var]
        time_dim = find_time_dim(da, lat_name, lon_name) if lat_name and lon_name else None

        print(f"file: {file_path}")
        print(f"sizes: {dict(ds.sizes)}")
        print(f"variables: {list(ds.data_vars)}")
        print(f"lat: {lat_name}")
        print(f"lon: {lon_name}")
        print(f"sst variable: {sst_var}")
        print(f"time dim: {time_dim}")

        sample = da.isel({time_dim: 0}) if time_dim else da
        sst = to_celsius(sample.values)
        finite = sst[np.isfinite(sst)]
        if finite.size:
            print(f"sst range degC: {float(np.nanmin(finite)):.3f} - {float(np.nanmax(finite)):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GHRSST/MUR-JPL NetCDF files.")
    parser.add_argument("input", type=Path, help="NetCDF file or directory")
    parser.add_argument("--max-files", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = [args.input] if args.input.is_file() else list_nc_files(args.input)
    for file_path in files[: args.max_files]:
        describe_file(file_path)
        print()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from read_ghrsst import guess_name, list_nc_files


def crop_dataset(ds: xr.Dataset, lon_range: tuple[float, float], lat_range: tuple[float, float]) -> xr.Dataset:
    lon_name = guess_name(ds, ("lon", "longitude", "x"))
    lat_name = guess_name(ds, ("lat", "latitude", "y"))
    if lon_name is None or lat_name is None:
        raise ValueError("Cannot identify latitude/longitude coordinates")

    lon_min, lon_max = min(lon_range), max(lon_range)
    lat_min, lat_max = min(lat_range), max(lat_range)
    cropped = ds.where((ds[lon_name] >= lon_min) & (ds[lon_name] <= lon_max), drop=True)
    cropped = cropped.where((cropped[lat_name] >= lat_min) & (cropped[lat_name] <= lat_max), drop=True)
    return cropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop NetCDF SST files to the South China Sea region.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/south_china_sea"))
    parser.add_argument("--lon-min", type=float, default=105.0)
    parser.add_argument("--lon-max", type=float, default=120.0)
    parser.add_argument("--lat-min", type=float, default=0.0)
    parser.add_argument("--lat-max", type=float, default=20.0)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = list_nc_files(args.input_dir)
    if args.max_files:
        files = files[: args.max_files]

    for index, file_path in enumerate(files, start=1):
        output_path = args.output_dir / file_path.name
        print(f"[{index}/{len(files)}] {file_path.name}")
        with xr.open_dataset(file_path) as ds:
            cropped = crop_dataset(ds, (args.lon_min, args.lon_max), (args.lat_min, args.lat_max))
            cropped.to_netcdf(output_path)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()

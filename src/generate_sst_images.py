from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from read_ghrsst import find_sst_var, find_time_dim, guess_name, list_nc_files, to_celsius


def safe_text(value: object) -> str:
    text = str(value).replace(".000000000", "")
    for char in (" ", ":", "/", "\\", "."):
        text = text.replace(char, "-")
    return text.replace("T", "_")


def save_sst_image(sst: np.ndarray, output_path: Path, title: str = "") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = sst[np.isfinite(sst)]
    vmin = float(np.nanpercentile(valid, 2)) if valid.size else None
    vmax = float(np.nanpercentile(valid, 98)) if valid.size else None

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    image = ax.imshow(np.ma.masked_invalid(sst), cmap="turbo", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="SST (degC)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_file(file_path: Path, output_dir: Path) -> int:
    saved = 0
    with xr.open_dataset(file_path) as ds:
        lat_name = guess_name(ds, ("lat", "latitude", "y"))
        lon_name = guess_name(ds, ("lon", "longitude", "x"))
        sst_var = find_sst_var(ds, lat_name, lon_name)
        da = ds[sst_var]
        time_dim = find_time_dim(da, lat_name, lon_name) if lat_name and lon_name else None

        if time_dim is None:
            sst = to_celsius(da.values)
            save_sst_image(sst, output_dir / f"{file_path.stem}_single.png", file_path.name)
            return 1

        for time_index in range(da.sizes[time_dim]):
            slice_da = da.isel({time_dim: time_index})
            time_label = safe_text(slice_da[time_dim].values)
            sst = to_celsius(slice_da.values)
            save_sst_image(sst, output_dir / f"{file_path.stem}_{time_label}.png", f"{file_path.name} {time_label}")
            saved += 1
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SST NetCDF slices as PNG images.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/interim/south_china_sea"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/sst_images"))
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = list_nc_files(args.input_dir)
    if args.max_files:
        files = files[: args.max_files]

    total = 0
    for index, file_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {file_path.name}")
        total += process_file(file_path, args.output_dir)
    print(f"saved images: {total}")


if __name__ == "__main__":
    main()

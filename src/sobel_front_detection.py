from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import ndimage

from read_ghrsst import find_sst_var, find_time_dim, guess_name, list_nc_files, to_celsius


def smooth_sst(sst: np.ndarray, valid_mask: np.ndarray, sigma: float) -> np.ndarray:
    values = np.where(valid_mask, sst, 0.0)
    weights = valid_mask.astype(float)
    weighted_sum = ndimage.gaussian_filter(values, sigma=sigma, mode="constant", cval=0.0)
    weight_sum = ndimage.gaussian_filter(weights, sigma=sigma, mode="constant", cval=0.0)
    smoothed = np.full_like(sst, np.nan, dtype=float)
    usable = weight_sum > 1e-6
    smoothed[usable] = weighted_sum[usable] / weight_sum[usable]
    smoothed[~valid_mask] = np.nan
    return smoothed


def sobel_gradient(smoothed: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float) / 8.0
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float) / 8.0
    filled = np.where(valid_mask, smoothed, 0.0)
    gx = ndimage.convolve(filled, kernel_x, mode="constant", cval=0.0)
    gy = ndimage.convolve(filled, kernel_y, mode="constant", cval=0.0)
    gradient = np.hypot(gx, gy)
    support = ndimage.convolve(valid_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant")
    gradient[(support != 9) | ~valid_mask] = np.nan
    return gradient


def detect_front_mask(gradient: np.ndarray, percentile: float, min_pixels: int) -> tuple[np.ndarray, float | None]:
    valid_gradient = gradient[np.isfinite(gradient)]
    if valid_gradient.size == 0:
        return np.zeros_like(gradient, dtype=bool), None
    threshold = float(np.nanpercentile(valid_gradient, percentile))
    raw_mask = (gradient >= threshold) & np.isfinite(gradient)
    labels, count = ndimage.label(raw_mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros_like(raw_mask, dtype=bool), threshold
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[labels], threshold


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(mask.astype(np.uint8), cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.axis("off")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_overlay(sst: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = sst[np.isfinite(sst)]
    vmin = float(np.nanpercentile(valid, 2)) if valid.size else None
    vmax = float(np.nanpercentile(valid, 98)) if valid.size else None
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(np.ma.masked_invalid(sst), cmap="turbo", origin="lower", vmin=vmin, vmax=vmax)
    ax.imshow(np.ma.masked_where(~mask, mask), cmap="Reds", origin="lower", alpha=0.65)
    ax.axis("off")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def safe_text(value: object) -> str:
    text = str(value).replace(".000000000", "")
    for char in (" ", ":", "/", "\\", "."):
        text = text.replace(char, "-")
    return text.replace("T", "_")


def process_file(file_path: Path, args: argparse.Namespace, writer: csv.writer) -> int:
    saved = 0
    with xr.open_dataset(file_path) as ds:
        lat_name = guess_name(ds, ("lat", "latitude", "y"))
        lon_name = guess_name(ds, ("lon", "longitude", "x"))
        sst_var = find_sst_var(ds, lat_name, lon_name)
        da = ds[sst_var]
        time_dim = find_time_dim(da, lat_name, lon_name) if lat_name and lon_name else None
        time_indices = range(da.sizes[time_dim]) if time_dim else [None]

        for time_index in time_indices:
            slice_da = da if time_index is None else da.isel({time_dim: time_index})
            time_label = "single" if time_index is None else safe_text(slice_da[time_dim].values)
            prefix = f"{file_path.stem}_{time_label}"
            sst = to_celsius(slice_da.values)
            valid_mask = np.isfinite(sst)
            smoothed = smooth_sst(sst, valid_mask, args.gaussian_sigma)
            gradient = sobel_gradient(smoothed, valid_mask)
            mask, threshold = detect_front_mask(gradient, args.gradient_percentile, args.min_front_pixels)

            mask_path = args.output_dir / "masks" / f"{prefix}_mask.png"
            overlay_path = args.output_dir / "overlays" / f"{prefix}_overlay.png"
            save_mask(mask, mask_path)
            save_overlay(sst, mask, overlay_path)
            writer.writerow([file_path.name, time_label, int(mask.sum()), threshold, mask_path, overlay_path])
            saved += 1
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect candidate SST fronts with Sobel gradients.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/interim/south_china_sea"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/sobel_fronts"))
    parser.add_argument("--gaussian-sigma", type=float, default=1.2)
    parser.add_argument("--gradient-percentile", type=float, default=97.5)
    parser.add_argument("--min-front-pixels", type=int, default=50)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = list_nc_files(args.input_dir)
    if args.max_files:
        files = files[: args.max_files]

    metadata_path = args.output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_file", "time", "front_pixels", "threshold", "mask_path", "overlay_path"])
        total = 0
        for index, file_path in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] {file_path.name}")
            total += process_file(file_path, args, writer)
    print(f"saved front masks: {total}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()

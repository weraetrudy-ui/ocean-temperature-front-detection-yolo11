from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    if not pairs:
        raise FileNotFoundError("No image/label pairs found")
    return pairs


def copy_split(pairs: list[tuple[Path, Path]], output_dir: Path, split_name: str) -> None:
    image_out = output_dir / "images" / split_name
    label_out = output_dir / "labels" / split_name
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_out / image_path.name)
        shutil.copy2(label_path, label_out / label_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split YOLO image/label pairs into train/val/test folders.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-ratio", type=float, default=0.74)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = collect_pairs(args.image_dir, args.label_dir)
    random.Random(args.seed).shuffle(pairs)

    train_end = int(len(pairs) * args.train_ratio)
    val_end = train_end + int(len(pairs) * args.val_ratio)
    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    for split_name, split_pairs in splits.items():
        copy_split(split_pairs, args.output_dir, split_name)
        print(f"{split_name}: {len(split_pairs)}")


if __name__ == "__main__":
    main()

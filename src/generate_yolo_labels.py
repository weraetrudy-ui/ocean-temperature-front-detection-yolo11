from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def mask_to_boxes(mask_path: Path, min_area: int) -> list[tuple[float, float, float, float]]:
    image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(mask_path)
    binary = (image > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary.shape
    boxes: list[tuple[float, float, float, float]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append(((x + w / 2) / width, (y + h / 2) / height, w / width, h / height))
    return boxes


def write_label(boxes: list[tuple[float, float, float, float]], output_path: Path, class_id: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in boxes]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert binary front masks to YOLO bounding-box labels.")
    parser.add_argument("--mask-dir", type=Path, default=Path("data/interim/sobel_fronts/masks"))
    parser.add_argument("--label-dir", type=Path, default=Path("data/interim/yolo_labels"))
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    masks = sorted(args.mask_dir.glob("*.png"))
    if not masks:
        raise FileNotFoundError(f"No mask PNG files found in {args.mask_dir}")

    for mask_path in masks:
        boxes = mask_to_boxes(mask_path, args.min_area)
        output_name = mask_path.name.replace("_mask", "")
        write_label(boxes, args.label_dir / f"{Path(output_name).stem}.txt", args.class_id)
    print(f"labels saved: {args.label_dir}")


if __name__ == "__main__":
    main()

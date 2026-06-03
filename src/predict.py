from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO prediction on SST front images.")
    parser.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    parser.add_argument("--source", type=Path, default=Path("data/samples/images"))
    parser.add_argument("--project", type=Path, default=Path("results/detection_examples"))
    parser.add_argument("--name", type=str, default="predict")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing weights: {args.weights}")
    if not args.source.exists():
        raise FileNotFoundError(f"Missing source: {args.source}")

    model = YOLO(args.weights)
    model.predict(source=args.source, project=args.project, name=args.name, conf=args.conf, save=True, exist_ok=True)


if __name__ == "__main__":
    main()

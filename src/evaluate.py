from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    parser.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--project", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--name", type=str, default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing weights: {args.weights}")

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, project=args.project, name=args.name, exist_ok=True, workers=0)
    print("precision:", metrics.box.mp)
    print("recall:", metrics.box.mr)
    print("mAP@0.5:", metrics.box.map50)
    print("mAP@0.5:0.95:", metrics.box.map)


if __name__ == "__main__":
    main()

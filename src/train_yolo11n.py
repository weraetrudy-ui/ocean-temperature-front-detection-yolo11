from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11n for ocean thermal front detection.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_yolo11n.yaml"))
    parser.add_argument("--weights", type=str, default=None, help="Optional initial weights, for example yolo11n.pt")
    parser.add_argument("--data", type=Path, default=None, help="Optional dataset yaml override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_name = args.weights or cfg.pop("model")
    if args.data is not None:
        cfg["data"] = str(args.data)

    model = YOLO(model_name)
    model.train(**cfg)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations
import argparse


def main():
    ap = argparse.ArgumentParser(description="Export YOLO model to ONNX/TensorRT via Ultralytics")
    ap.add_argument("--model", required=True, help="Path to .pt weights")
    ap.add_argument("--format", default="onnx", choices=["onnx", "engine", "openvino", "torchscript"], help="Export format")
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Ultralytics not installed: ", e)
        return

    model = YOLO(args.model)
    model.export(format=args.format, imgsz=args.imgsz)
    print(f"Exported {args.model} to {args.format}")


if __name__ == "__main__":
    main()


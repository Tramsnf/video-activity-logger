#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <data.yaml> [model=yolov8n.pt] [epochs=100] [name=exp]"
  exit 1
fi

DATA="$1"; MODEL="${2:-yolov8n.pt}"; EPOCHS="${3:-100}"; NAME="${4:-vap-exp}"

# Requires ultralytics installed in the current Python environment
yolo detect train model="$MODEL" data="$DATA" epochs="$EPOCHS" imgsz=960 batch=-1 project=runs/vap name="$NAME"

echo "Training complete. Weights at runs/vap/$NAME/weights/best.pt"


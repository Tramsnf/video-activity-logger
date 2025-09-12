#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
from typing import Dict, Tuple, List, Any
import math
import pandas as pd

from vap.config import load_config
from vap.detect import build_detector
from vap.track import build_tracker, Track
from vap import ingest


def centroid(xyxy):
    x1, y1, x2, y2 = xyxy
    return ( (x1 + x2) / 2.0, (y1 + y2) / 2.0 )


def main():
    ap = argparse.ArgumentParser(description="Export per-actor time-series features for training")
    ap.add_argument("--config", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="Output features file (.parquet or .csv)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fps = ingest.probe_fps(args.video) or cfg.thresholds.fps_target
    det = build_detector(
        cfg.detect.backend,
        cfg.detect.model_path,
        cfg.detect.classes,
        min_conf=cfg.detect.min_conf,
        min_box_area=cfg.detect.min_box_area,
        imgsz=cfg.detect.imgsz,
        device=cfg.detect.device,
        fp16=cfg.detect.fp16,
    )
    trk = build_tracker(cfg.track.backend, max_age=cfg.track.max_age)

    # State buffers
    last_pos: Dict[str, Tuple[float, float]] = {}
    last_speed: Dict[str, float] = {}
    last_dir: Dict[str, float] = {}

    rows: List[Dict[str, Any]] = []
    batch = []
    batch_size = max(1, int(getattr(cfg.detect, "batch_size", 1)))

    def process(frames_batch):
        nonlocal rows, last_pos, last_speed, last_dir
        if not frames_batch:
            return
        idxs, frames = zip(*frames_batch)
        all_dets = det.infer(frames) if hasattr(det, "infer") else [det(f) for f in frames]
        for fi, dets in zip(idxs, all_dets):
            tracks: List[Track] = trk.update(dets)
            # Precompute nearest pallet distance per actor
            pallet_centers = [centroid(t.xyxy) for t in tracks if t.cls_name == "pallet"]
            for t in tracks:
                aid = f"{t.cls_name}_{t.track_id}"
                cx, cy = centroid(t.xyxy)
                # size features
                x1, y1, x2, y2 = t.xyxy
                w, h = max(1e-3, x2 - x1), max(1e-3, y2 - y1)
                area = w * h
                # motion features
                speed = 0.0
                direction = last_dir.get(aid, 0.0)
                if aid in last_pos:
                    dx = cx - last_pos[aid][0]
                    dy = cy - last_pos[aid][1]
                    speed = math.hypot(dx, dy) * fps / cfg.thresholds.pixels_per_meter
                    direction = math.atan2(dy, dx)
                acc = 0.0
                if aid in last_speed:
                    acc = (speed - last_speed[aid]) * fps
                dir_change = 0.0
                if aid in last_dir:
                    d = direction - last_dir[aid]
                    while d > math.pi: d -= 2 * math.pi
                    while d < -math.pi: d += 2 * math.pi
                    dir_change = abs(d)
                # nearest pallet distance
                nearest_pallet_m = None
                if pallet_centers:
                    nearest_pallet_m = min(
                        math.hypot(cx - px, cy - py) for px, py in pallet_centers
                    ) / cfg.thresholds.pixels_per_meter
                rows.append({
                    "video_id": cfg.video_id,
                    "frame_idx": fi,
                    "time_s": fi / fps,
                    "actor_id": aid,
                    "actor_type": t.cls_name,
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h,
                    "area": area,
                    "speed_mps": speed,
                    "acc_mps2": acc,
                    "dir_rad": direction,
                    "dir_change_rad": dir_change,
                    "nearest_pallet_dist_m": nearest_pallet_m,
                })
                last_pos[aid] = (cx, cy)
                last_speed[aid] = speed
                last_dir[aid] = direction

    for idx, frame in ingest.frames(args.video):
        batch.append((idx, frame))
        if len(batch) >= batch_size:
            process(batch)
            batch = []
    if batch:
        process(batch)

    df = pd.DataFrame(rows)
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if out.endswith(".parquet"):
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    print(f"Wrote features → {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()


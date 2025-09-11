from __future__ import annotations
import argparse, os
from .config import load_config
from .taxonomy import load_taxonomy
from . import ingest
from .detect import build_detector
from .track import build_tracker
from .states import update_state_machine, close_open_states, postprocess_state_events
from .actions import build_action_heuristics
from .io import write_events
import cv2, numpy as np

def main():
    ap = argparse.ArgumentParser(description="Video Activity Logger")
    ap.add_argument("--config", required=True, help="Path to pipeline YAML")
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out", default="outputs", help="Output directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tax = load_taxonomy(cfg.taxonomy_path)
    # Optional ROI mask: if provided in config, load as binary mask
    roi = None
    roi_path = getattr(cfg, "roi_mask_path", None)
    if roi_path:
        roi_img = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        roi = (roi_img > 0).astype("uint8") if roi_img is not None else None

    det = build_detector(cfg.detect.backend, cfg.detect.model_path, cfg.detect.classes)
    # if YOLO, set detector’s filters/roi (quick way without refactor)
    if hasattr(det, "min_conf"): 
        det.min_conf = getattr(cfg.detect, "min_conf", 0.4)
        det.min_box_area = getattr(cfg.detect, "min_box_area", 900)
        if roi is not None: det.roi = roi
        
    trk = build_tracker(cfg.track.backend)
    act = build_action_heuristics()

    actors = {}
    events = []
    frame_idx = 0

    # Probe FPS from video if possible
    import cv2
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.thresholds.fps_target
    cap.release()

    # Resolve thresholds (backward-compatible with old schema)
    thr = cfg.thresholds
    # Backward-compatibility: fall back to old speed_* if present
    fl_speed_drive = getattr(thr, "fl_speed_drive", getattr(thr, "speed_drive", 0.2))
    fl_speed_stop  = getattr(thr, "fl_speed_stop",  getattr(thr, "speed_stop",  0.05))
    hu_speed_walk  = getattr(thr, "hu_speed_walk", fl_speed_drive)
    hu_speed_wait  = getattr(thr, "hu_speed_wait",  fl_speed_stop)

    for frame_idx, frame in ingest.frames(args.video):
        detections = det(frame)
        tracks = trk.update(detections)
        ev = update_state_machine(
            actors=actors, tracks=tracks, frame_idx=frame_idx, fps=fps,
            pixels_per_meter=cfg.thresholds.pixels_per_meter,
            fl_speed_drive=fl_speed_drive,
            fl_speed_stop=fl_speed_stop,
            hu_speed_walk=hu_speed_walk,
            hu_speed_wait=hu_speed_wait,
            debounce_frames=cfg.thresholds.debounce_frames,
            source_camera=cfg.source_camera, video_id=cfg.video_id
        )
        if ev: events.extend(ev)

        # Action heuristics (GRAB/PLACE)
        ev2 = act.update(
            actors=actors,
            tracks=tracks,
            frame_idx=frame_idx,
            fps=fps,
            pixels_per_meter=cfg.thresholds.pixels_per_meter,
            thresholds=thr,
            source_camera=cfg.source_camera,
            video_id=cfg.video_id,
        )
        if ev2: events.extend(ev2)

    events.extend(close_open_states(actors, frame_idx, fps, cfg.source_camera, cfg.video_id))
    # Post-process state events: merge tiny gaps, drop too-short intervals
    min_dur = float(getattr(thr, "min_state_dur_s", 0.0))
    merge_gap = float(getattr(thr, "merge_gap_s", 0.0))
    if min_dur > 0 or merge_gap > 0:
        events = postprocess_state_events(events, tax, min_dur, merge_gap)
    out_csv = write_events(args.out, events, stem=f"{cfg.video_id}_events")
    print(f"Wrote {len(events)} events → {out_csv}")

if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse, os
from .config import load_config
from .taxonomy import load_taxonomy
from . import ingest
from .detect import build_detector
from .track import build_tracker
from .states import update_state_machine, close_open_states, postprocess_state_events
from .actions import build_action_heuristics
from .zones import Zones, ZoneEventer
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

    det = build_detector(
        cfg.detect.backend,
        cfg.detect.model_path,
        cfg.detect.classes,
        min_conf=cfg.detect.min_conf,
        min_box_area=cfg.detect.min_box_area,
        roi_mask=roi,
        imgsz=cfg.detect.imgsz,
        device=cfg.detect.device,
        fp16=cfg.detect.fp16,
    )
    # if YOLO, set detector’s filters/roi (quick way without refactor)
    if hasattr(det, "min_conf"):
        det.min_conf = cfg.detect.min_conf
        det.min_box_area = cfg.detect.min_box_area
        if roi is not None:
            det.roi = roi
        
    trk = build_tracker(cfg.track.backend, max_age=cfg.track.max_age)
    act = build_action_heuristics()
    # Zones
    zone_eventer = None
    if getattr(cfg, "zones_path", None):
        try:
            zones = Zones.load(cfg.zones_path)
            zone_eventer = ZoneEventer(zones)
        except Exception as e:
            print(f"Warning: failed to load zones from {cfg.zones_path}: {e}")

    actors = {}
    events = []
    frame_idx = 0

    # Probe FPS from video if possible
    from . import ingest as _ing
    fps = _ing.probe_fps(args.video) or cfg.thresholds.fps_target

    # Resolve thresholds (backward-compatible with old schema)
    thr = cfg.thresholds
    # Backward-compatibility: fall back to old speed_* if present
    fl_speed_drive = getattr(thr, "fl_speed_drive", getattr(thr, "speed_drive", 0.2))
    fl_speed_stop  = getattr(thr, "fl_speed_stop",  getattr(thr, "speed_stop",  0.05))
    hu_speed_walk  = getattr(thr, "hu_speed_walk", fl_speed_drive)
    hu_speed_wait  = getattr(thr, "hu_speed_wait",  fl_speed_stop)

    batch_size = max(1, int(getattr(cfg.detect, "batch_size", 1)))
    frame_buffer = []  # list of (idx, frame)
    for frame_idx, frame in ingest.frames(args.video):
        frame_buffer.append((frame_idx, frame))
        if len(frame_buffer) >= batch_size:
            idxs, frames = zip(*frame_buffer)
            all_dets = det.infer(frames) if hasattr(det, "infer") else [det(f) for f in frames]
            for fi, dets in zip(idxs, all_dets):
                tracks = trk.update(dets)
                ev = update_state_machine(
                    actors=actors, tracks=tracks, frame_idx=fi, fps=fps,
                    pixels_per_meter=cfg.thresholds.pixels_per_meter,
                    fl_speed_drive=fl_speed_drive,
                    fl_speed_stop=fl_speed_stop,
                    hu_speed_walk=hu_speed_walk,
                    hu_speed_wait=hu_speed_wait,
                    debounce_frames=cfg.thresholds.debounce_frames,
                    source_camera=cfg.source_camera, video_id=cfg.video_id
                )
                if ev: events.extend(ev)
                # Actions
                ev2 = act.update(
                    actors=actors,
                    tracks=tracks,
                    frame_idx=fi,
                    fps=fps,
                    pixels_per_meter=cfg.thresholds.pixels_per_meter,
                    thresholds=thr,
                    source_camera=cfg.source_camera,
                    video_id=cfg.video_id,
                )
                if ev2: events.extend(ev2)
                # Zones
                if zone_eventer is not None:
                    ev3 = zone_eventer.update(tracks, fi, fps, cfg.source_camera, cfg.video_id)
                    if ev3: events.extend(ev3)
            frame_buffer = []
    # Flush remainder
    if frame_buffer:
        idxs, frames = zip(*frame_buffer)
        all_dets = det.infer(frames) if hasattr(det, "infer") else [det(f) for f in frames]
        for fi, dets in zip(idxs, all_dets):
            tracks = trk.update(dets)
            ev = update_state_machine(
                actors=actors, tracks=tracks, frame_idx=fi, fps=fps,
                pixels_per_meter=cfg.thresholds.pixels_per_meter,
                fl_speed_drive=fl_speed_drive,
                fl_speed_stop=fl_speed_stop,
                hu_speed_walk=hu_speed_walk,
                hu_speed_wait=hu_speed_wait,
                debounce_frames=cfg.thresholds.debounce_frames,
                source_camera=cfg.source_camera, video_id=cfg.video_id
            )
            if ev: events.extend(ev)
            ev2 = act.update(
                actors=actors,
                tracks=tracks,
                frame_idx=fi,
                fps=fps,
                pixels_per_meter=cfg.thresholds.pixels_per_meter,
                thresholds=thr,
                source_camera=cfg.source_camera,
                video_id=cfg.video_id,
            )
            if ev2: events.extend(ev2)
            if zone_eventer is not None:
                ev3 = zone_eventer.update(tracks, fi, fps, cfg.source_camera, cfg.video_id)
                if ev3: events.extend(ev3)

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

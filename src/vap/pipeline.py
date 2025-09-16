from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
import tempfile

import cv2
import numpy as np

from .config import PipelineConfig
from .taxonomy import load_taxonomy
from .detect import build_detector, Detection
from .track import build_tracker, Track
from .states import update_state_machine, close_open_states, postprocess_state_events
from .actions import build_action_heuristics
from .zones import Zones, ZoneEventer
from . import ingest
from .io import write_events


ProgressCallback = Callable[[int, Optional[int]], None]
FrameCallback = Callable[[int, np.ndarray], None]


@dataclass
class FrameStat:
    frame_idx: int
    num_detections: int
    num_tracks: int


@dataclass
class PipelineResult:
    events: List[Dict[str, Any]]
    event_csv_path: Optional[str]
    annotated_video_path: Optional[str]
    fps: float
    frame_count: int
    frame_stats: List[FrameStat]


def _apply_detect_overrides(cfg: PipelineConfig, overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    if not overrides:
        return cfg
    cfg_copy = cfg.model_copy(deep=True)
    for key, value in overrides.items():
        if hasattr(cfg_copy.detect, key):
            setattr(cfg_copy.detect, key, value)
    return cfg_copy


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    palette = (
        (231, 76, 60),
        (46, 204, 113),
        (52, 152, 219),
        (155, 89, 182),
        (241, 196, 15),
        (26, 188, 156),
        (230, 126, 34),
        (127, 140, 141),
    )
    return palette[track_id % len(palette)]


def _draw_annotations(frame: np.ndarray, detections: List[Detection], tracks: List[Track]) -> np.ndarray:
    annotated = frame.copy()
    # Draw detections (thin boxes)
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 200, 200), 1)
        label = f"{det.cls_name}:{det.conf:.2f}"
        cv2.putText(annotated, label, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    # Draw tracks (thick boxes + id)
    for tr in tracks:
        x1, y1, x2, y2 = map(int, tr.xyxy)
        color = _color_for_track(tr.track_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"ID {tr.track_id}"
        if tr.cls_name:
            label = f"{label} · {tr.cls_name}"
        cv2.putText(annotated, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return annotated


def analyze_video(
    cfg: PipelineConfig,
    video_path: str,
    *,
    detection_overrides: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str | Path] = None,
    progress_callback: Optional[ProgressCallback] = None,
    frame_callback: Optional[FrameCallback] = None,
    create_annotated_video: bool = False,
    annotation_path: Optional[str | Path] = None,
    max_frames: Optional[int] = None,
) -> PipelineResult:
    """Run the activity pipeline for a video and optionally export artifacts."""

    src_cfg = _apply_detect_overrides(cfg, detection_overrides)

    cfg_local = src_cfg if src_cfg is not cfg else cfg.model_copy(deep=True)

    tax = load_taxonomy(cfg_local.taxonomy_path)

    roi = None
    roi_path = getattr(cfg_local, "roi_mask_path", None)
    if roi_path:
        roi_img = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        roi = (roi_img > 0).astype("uint8") if roi_img is not None else None

    det = build_detector(
        cfg_local.detect.backend,
        cfg_local.detect.model_path,
        cfg_local.detect.classes,
        min_conf=cfg_local.detect.min_conf,
        min_box_area=cfg_local.detect.min_box_area,
        roi_mask=roi,
        imgsz=cfg_local.detect.imgsz,
        device=cfg_local.detect.device,
        fp16=cfg_local.detect.fp16,
    )
    if hasattr(det, "min_conf"):
        det.min_conf = cfg_local.detect.min_conf
        det.min_box_area = cfg_local.detect.min_box_area
        if roi is not None:
            det.roi = roi

    tracker = build_tracker(cfg_local.track.backend, max_age=cfg_local.track.max_age)
    action_heuristics = build_action_heuristics()

    zone_eventer = None
    if getattr(cfg_local, "zones_path", None):
        try:
            zones = Zones.load(cfg_local.zones_path)
            zone_eventer = ZoneEventer(zones)
        except Exception:
            zone_eventer = None

    actors: Dict[int, Any] = {}
    events: List[Dict[str, Any]] = []
    frame_stats: List[FrameStat] = []

    fps = ingest.probe_fps(video_path) or cfg_local.thresholds.fps_target
    total_frames = ingest.frame_count(video_path)

    batch_size = max(1, int(getattr(cfg_local.detect, "batch_size", 1)))
    thr = cfg_local.thresholds
    fl_speed_drive = getattr(thr, "fl_speed_drive", getattr(thr, "speed_drive", 0.2))
    fl_speed_stop = getattr(thr, "fl_speed_stop", getattr(thr, "speed_stop", 0.05))
    hu_speed_walk = getattr(thr, "hu_speed_walk", fl_speed_drive)
    hu_speed_wait = getattr(thr, "hu_speed_wait", fl_speed_stop)
    frame_buffer: List[tuple[int, np.ndarray]] = []
    video_writer = None
    annotated_path: Optional[str] = None
    latest_idx = 0

    def _ensure_writer(frame: np.ndarray) -> None:
        nonlocal video_writer, annotated_path
        if not create_annotated_video or video_writer is not None:
            return
        path = annotation_path
        if path is None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.close()
            path = tmp.name
        annotated_path = str(path)
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(annotated_path, fourcc, fps or 30.0, (w, h))

    for frame_idx, frame in ingest.frames(video_path):
        latest_idx = frame_idx
        frame_buffer.append((frame_idx, frame))
        if len(frame_buffer) < batch_size:
            if progress_callback:
                progress_callback(frame_idx, total_frames)
            if max_frames and frame_idx >= max_frames:
                break
            continue

        idxs, frames = zip(*frame_buffer)
        all_dets = det.infer(frames) if hasattr(det, "infer") else [det(f) for f in frames]
        for fi, fr, dets in zip(idxs, frames, all_dets):
            tracks = tracker.update(dets)
            frame_stats.append(FrameStat(frame_idx=fi, num_detections=len(dets), num_tracks=len(tracks)))

            ev = update_state_machine(
                actors=actors,
                tracks=tracks,
                frame_idx=fi,
                fps=fps,
                pixels_per_meter=cfg_local.thresholds.pixels_per_meter,
                fl_speed_drive=fl_speed_drive,
                fl_speed_stop=fl_speed_stop,
                hu_speed_walk=hu_speed_walk,
                hu_speed_wait=hu_speed_wait,
                debounce_frames=cfg_local.thresholds.debounce_frames,
                source_camera=cfg_local.source_camera,
                video_id=cfg_local.video_id,
            )
            if ev:
                events.extend(ev)

            ev2 = action_heuristics.update(
                actors=actors,
                tracks=tracks,
                frame_idx=fi,
                fps=fps,
                pixels_per_meter=cfg_local.thresholds.pixels_per_meter,
                thresholds=cfg_local.thresholds,
                source_camera=cfg_local.source_camera,
                video_id=cfg_local.video_id,
            )
            if ev2:
                events.extend(ev2)

            if zone_eventer is not None:
                ev3 = zone_eventer.update(tracks, fi, fps, cfg_local.source_camera, cfg_local.video_id)
                if ev3:
                    events.extend(ev3)

            annotated_frame: Optional[np.ndarray] = None
            should_annotate = create_annotated_video or frame_callback is not None
            if should_annotate:
                annotated_frame = _draw_annotations(fr, dets, tracks)

            if create_annotated_video:
                _ensure_writer(fr)
                if video_writer is not None and annotated_frame is not None:
                    video_writer.write(annotated_frame)

            if frame_callback is not None:
                frame_callback(fi, annotated_frame if annotated_frame is not None else fr)

            if progress_callback:
                progress_callback(fi, total_frames)

            if max_frames and fi >= max_frames:
                break
        frame_buffer = []
        if max_frames and latest_idx >= max_frames:
            break

    if frame_buffer and (not max_frames or latest_idx < max_frames):
        idxs, frames = zip(*frame_buffer)
        all_dets = det.infer(frames) if hasattr(det, "infer") else [det(f) for f in frames]
        for fi, fr, dets in zip(idxs, frames, all_dets):
            tracks = tracker.update(dets)
            frame_stats.append(FrameStat(frame_idx=fi, num_detections=len(dets), num_tracks=len(tracks)))

            ev = update_state_machine(
                actors=actors,
                tracks=tracks,
                frame_idx=fi,
                fps=fps,
                pixels_per_meter=cfg_local.thresholds.pixels_per_meter,
                fl_speed_drive=fl_speed_drive,
                fl_speed_stop=fl_speed_stop,
                hu_speed_walk=hu_speed_walk,
                hu_speed_wait=hu_speed_wait,
                debounce_frames=cfg_local.thresholds.debounce_frames,
                source_camera=cfg_local.source_camera,
                video_id=cfg_local.video_id,
            )
            if ev:
                events.extend(ev)

            ev2 = action_heuristics.update(
                actors=actors,
                tracks=tracks,
                frame_idx=fi,
                fps=fps,
                pixels_per_meter=cfg_local.thresholds.pixels_per_meter,
                thresholds=cfg_local.thresholds,
                source_camera=cfg_local.source_camera,
                video_id=cfg_local.video_id,
            )
            if ev2:
                events.extend(ev2)

            if zone_eventer is not None:
                ev3 = zone_eventer.update(tracks, fi, fps, cfg_local.source_camera, cfg_local.video_id)
                if ev3:
                    events.extend(ev3)

            annotated_frame = None
            should_annotate = create_annotated_video or frame_callback is not None
            if should_annotate:
                annotated_frame = _draw_annotations(fr, dets, tracks)

            if create_annotated_video:
                _ensure_writer(fr)
                if video_writer is not None and annotated_frame is not None:
                    video_writer.write(annotated_frame)

            if frame_callback is not None:
                frame_callback(fi, annotated_frame if annotated_frame is not None else fr)

            if progress_callback:
                progress_callback(fi, total_frames)

            if max_frames and fi >= max_frames:
                break

    if video_writer is not None:
        video_writer.release()

    latest_idx = max(latest_idx, frame_stats[-1].frame_idx if frame_stats else 0)

    events.extend(close_open_states(actors, latest_idx, fps, cfg_local.source_camera, cfg_local.video_id))

    min_dur = float(getattr(cfg_local.thresholds, "min_state_dur_s", 0.0))
    merge_gap = float(getattr(cfg_local.thresholds, "merge_gap_s", 0.0))
    if min_dur > 0 or merge_gap > 0:
        events = postprocess_state_events(events, tax, min_dur, merge_gap)

    event_csv_path = None
    if output_dir:
        event_csv_path = write_events(str(output_dir), events, stem=f"{cfg_local.video_id}_events")

    if progress_callback:
        progress_callback(latest_idx if latest_idx else (total_frames or 0), total_frames)

    return PipelineResult(
        events=events,
        event_csv_path=event_csv_path,
        annotated_video_path=annotated_path,
        fps=float(fps),
        frame_count=latest_idx,
        frame_stats=frame_stats,
    )

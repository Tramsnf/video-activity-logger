from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any, Iterator, Tuple
import contextlib
import queue
import tempfile
import threading

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
    total_frames: Optional[int]
    frame_stats: List[FrameStat]


class _FramePrefetcher:
    """Background reader that keeps a queue of decoded frames."""

    def __init__(self, video_path: str, maxsize: int) -> None:
        self._video_path = video_path
        self._queue: "queue.Queue[object]" = queue.Queue(max(1, maxsize))
        self._sentinel = object()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._exc: Optional[BaseException] = None
        self._thread.start()

    def _worker(self) -> None:
        try:
            for item in ingest.frames(self._video_path):
                if self._stop.is_set():
                    break
                while not self._stop.is_set():
                    try:
                        self._queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # pragma: no cover - propagate to consumer
            self._exc = exc
        finally:
            # Always signal completion (or error) to consumer
            while True:
                try:
                    self._queue.put(self._sentinel, timeout=0.1)
                    break
                except queue.Full:
                    try:
                        _ = self._queue.get_nowait()
                    except queue.Empty:
                        continue

    def __iter__(self) -> "_FramePrefetcher":
        return self

    def __next__(self) -> Tuple[int, np.ndarray]:
        item = self._queue.get()
        if item is self._sentinel:
            self.close()
            if self._exc:
                raise self._exc
            raise StopIteration
        return item  # type: ignore[return-value]

    def close(self) -> None:
        if not self._stop.is_set():
            self._stop.set()
        while self._thread.is_alive():
            try:
                self._queue.put_nowait(self._sentinel)
            except queue.Full:
                try:
                    _ = self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._thread.join(timeout=0.1)


@contextlib.contextmanager
def _frame_iterator(video_path: str, prefetch: int) -> Iterator[Iterator[Tuple[int, np.ndarray]]]:
    """Yield an iterator over frames, optionally prefetching them."""

    if prefetch <= 0:
        yield ingest.frames(video_path)
        return

    prefetcher = _FramePrefetcher(video_path, maxsize=prefetch)
    try:
        yield iter(prefetcher)
    finally:
        prefetcher.close()


def _resolve_device(device: Optional[str]) -> Optional[str]:
    """Return a valid device string for Ultralytics/torch."""

    if not device:
        return None

    choice = device.strip()
    if not choice:
        return None

    lowered = choice.lower()
    if lowered in {"auto", "default"}:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                return "0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return "mps"
            return "cpu"
        except Exception:
            return "cpu"

    if lowered in {"cpu"}:
        return "cpu"
    if lowered in {"mps"}:  # Apple Metal support
        return choice

    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"

    if lowered == "cuda":
        return "0" if torch.cuda.is_available() else "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    def _valid_cuda_index(idx: str) -> bool:
        try:
            num = int(idx)
        except ValueError:
            return False
        return 0 <= num < torch.cuda.device_count()

    if "," in choice:
        parts = [p.strip() for p in choice.split(",") if p.strip()]
        if parts and all(_valid_cuda_index(p) for p in parts):
            return ",".join(parts)
        return "cpu"

    if lowered.startswith("cuda:"):
        idx = lowered.split(":", 1)[1]
        return choice if _valid_cuda_index(idx) else "cpu"

    if choice.isdigit():
        return choice if _valid_cuda_index(choice) else "cpu"

    return choice


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

    cfg_local.detect.device = _resolve_device(getattr(cfg_local.detect, "device", None))

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
        max_det=cfg_local.detect.max_det,
        iou=cfg_local.detect.iou,
        agnostic_nms=cfg_local.detect.agnostic_nms,
    )
    if hasattr(det, "min_conf"):
        det.min_conf = cfg_local.detect.min_conf
        det.min_box_area = cfg_local.detect.min_box_area
        if roi is not None:
            det.roi = roi
        if hasattr(det, "iou"):
            det.iou = cfg_local.detect.iou
        if hasattr(det, "agnostic_nms"):
            det.agnostic_nms = cfg_local.detect.agnostic_nms
        if hasattr(det, "max_det"):
            det.max_det = cfg_local.detect.max_det

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
    detect_stride = max(1, int(getattr(cfg_local.detect, "stride", 1)))
    prefetch_frames = max(0, int(getattr(cfg_local.detect, "prefetch_frames", 0)))
    thr = cfg_local.thresholds
    fl_speed_drive = getattr(thr, "fl_speed_drive", getattr(thr, "speed_drive", 0.2))
    fl_speed_stop = getattr(thr, "fl_speed_stop", getattr(thr, "speed_stop", 0.05))
    hu_speed_walk = getattr(thr, "hu_speed_walk", fl_speed_drive)
    hu_speed_wait = getattr(thr, "hu_speed_wait", fl_speed_stop)
    pending_frames: List[tuple[int, np.ndarray, bool]] = []
    detection_frames: List[tuple[int, np.ndarray]] = []
    video_writer = None
    annotated_path: Optional[str] = None
    latest_idx = 0
    frame_shape: Optional[tuple[int, int]] = None
    max_track_age = max(1, int(getattr(cfg_local.track, "max_age", 30)))
    track_state: Dict[int, Dict[str, Any]] = {}

    def _update_track_state(tracks: List[Track], frame_idx: int) -> None:
        observed: set[int] = set()
        for tr in tracks:
            x1, y1, x2, y2 = tr.xyxy
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            prev = track_state.get(tr.track_id)
            if prev is not None:
                prev_cx, prev_cy = prev["center"]
                prev_idx = prev["frame_idx"]
                delta = max(1, frame_idx - prev_idx)
                raw_vx = (cx - prev_cx) / float(delta)
                raw_vy = (cy - prev_cy) / float(delta)
                old_vx, old_vy = prev["velocity"]
                smooth = 0.8
                vx = smooth * old_vx + (1.0 - smooth) * raw_vx
                vy = smooth * old_vy + (1.0 - smooth) * raw_vy
            else:
                vx = 0.0
                vy = 0.0
            track_state[tr.track_id] = {
                "cls": tr.cls_name,
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "size": (w, h),
                "velocity": (vx, vy),
                "frame_idx": frame_idx,
                "age": 0,
            }
            observed.add(tr.track_id)

        to_remove: List[int] = []
        for tid, state in track_state.items():
            if tid not in observed:
                state["age"] += 1
                if state["age"] > max_track_age:
                    to_remove.append(tid)
        for tid in to_remove:
            track_state.pop(tid, None)

    def _predict_tracks(frame_idx: int, frame_shape_local: Optional[tuple[int, int]]) -> List[Track]:
        if not track_state or frame_shape_local is None:
            return []
        height, width = frame_shape_local
        predicted: List[Track] = []
        to_remove: List[int] = []
        for tid, state in track_state.items():
            delta = frame_idx - state["frame_idx"]
            if delta <= 0:
                continue
            vx, vy = state["velocity"]
            cx, cy = state["center"]
            new_cx = cx + vx * delta
            new_cy = cy + vy * delta
            w, h = state["size"]
            x1 = max(0.0, new_cx - w * 0.5)
            y1 = max(0.0, new_cy - h * 0.5)
            x2 = min(float(width), new_cx + w * 0.5)
            y2 = min(float(height), new_cy + h * 0.5)
            if x2 <= x1 or y2 <= y1:
                state["age"] += delta
            else:
                state["center"] = (new_cx, new_cy)
                state["bbox"] = (x1, y1, x2, y2)
                state["frame_idx"] = frame_idx
                state["age"] += 1
                predicted.append(Track(track_id=tid, cls_name=state["cls"], xyxy=(x1, y1, x2, y2)))
        for tid in list(track_state.keys()):
            if track_state[tid]["age"] > max_track_age:
                track_state.pop(tid, None)
        return predicted

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

    def _run_detection(frames_for_detection: List[tuple[int, np.ndarray]]) -> Dict[int, List[Detection]]:
        if not frames_for_detection:
            return {}
        idxs, frames_batch = zip(*frames_for_detection)
        detections_batch = det.infer(frames_batch) if hasattr(det, "infer") else [det(f) for f in frames_batch]
        return {idx: dets for idx, dets in zip(idxs, detections_batch)}

    def _process_frame(
        frame_idx: int,
        frame: np.ndarray,
        detections: Optional[List[Detection]],
    ) -> None:
        nonlocal latest_idx, frame_shape
        latest_idx = frame_idx
        if frame_shape is None:
            frame_shape = frame.shape[:2]

        if detections is not None:
            tracks = tracker.update(detections)
            _update_track_state(tracks, frame_idx)
            det_count = len(detections)
        else:
            tracker.update([])
            tracks = _predict_tracks(frame_idx, frame_shape)
            det_count = 0

        frame_stats.append(
            FrameStat(frame_idx=frame_idx, num_detections=det_count, num_tracks=len(tracks))
        )

        ev = update_state_machine(
            actors=actors,
            tracks=tracks,
            frame_idx=frame_idx,
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
            frame_idx=frame_idx,
            fps=fps,
            pixels_per_meter=cfg_local.thresholds.pixels_per_meter,
            thresholds=cfg_local.thresholds,
            source_camera=cfg_local.source_camera,
            video_id=cfg_local.video_id,
        )
        if ev2:
            events.extend(ev2)

        if zone_eventer is not None:
            ev3 = zone_eventer.update(tracks, frame_idx, fps, cfg_local.source_camera, cfg_local.video_id)
            if ev3:
                events.extend(ev3)

        annotated_frame: Optional[np.ndarray] = None
        should_annotate = create_annotated_video or frame_callback is not None
        if should_annotate:
            dets_for_draw = detections if detections is not None else []
            annotated_frame = _draw_annotations(frame, dets_for_draw, tracks)

        if create_annotated_video:
            _ensure_writer(frame)
            if video_writer is not None and annotated_frame is not None:
                video_writer.write(annotated_frame)

        if frame_callback is not None:
            frame_callback(frame_idx, annotated_frame if annotated_frame is not None else frame)

        if progress_callback:
            progress_callback(frame_idx, total_frames)

    def _flush_pending() -> None:
        nonlocal pending_frames, detection_frames
        if not pending_frames:
            return
        detection_map = _run_detection(detection_frames)
        for fi, fr, need_det in pending_frames:
            dets = detection_map.get(fi) if need_det else None
            _process_frame(fi, fr, dets)
        pending_frames = []
        detection_frames = []

    with _frame_iterator(video_path, prefetch_frames) as frame_iter:
        for frame_idx, frame in frame_iter:
            request_detection = ((frame_idx - 1) % detect_stride == 0)
            pending_frames.append((frame_idx, frame, request_detection))
            if request_detection:
                detection_frames.append((frame_idx, frame))

            flush_due = False
            if detection_frames and len(detection_frames) >= batch_size:
                flush_due = True
            elif detection_frames and detect_stride > 1 and len(pending_frames) >= detect_stride:
                flush_due = True

            if max_frames and frame_idx >= max_frames:
                flush_due = True

            if flush_due:
                _flush_pending()
                if max_frames and frame_idx >= max_frames:
                    break

        # Process any remaining frames
        if pending_frames:
            _flush_pending()

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
        total_frames=total_frames if isinstance(total_frames, int) else None,
        frame_stats=frame_stats,
    )

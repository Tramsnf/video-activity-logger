from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.optimize import linear_sum_assignment

@dataclass
class Track:
    track_id: int
    cls_name: str
    xyxy: Tuple[float, float, float, float]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0:
        return 0.0
    a = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    b = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / max(1e-6, (a + b - inter))

class SimpleIOUTracker:
    """Matches detections to existing tracks by IoU; starts new tracks for unmatched dets."""
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}  # id -> {cls_name, xyxy, age}

    def update(self, detections) -> List[Track]:
        # Group detections by class
        dets_by_cls: Dict[str, List] = {}
        for d in detections:
            dets_by_cls.setdefault(d.cls_name, []).append(d)

        # Age tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1

        outputs: List[Track] = []

        # Match per class
        for cls_name, dets in dets_by_cls.items():
            active_ids = [tid for tid, t in self.tracks.items() if t["cls_name"] == cls_name]
            active_boxes = [self.tracks[tid]["xyxy"] for tid in active_ids]
            det_boxes = [tuple(d.xyxy) for d in dets]
            if active_boxes:
                cost = np.ones((len(active_boxes), len(det_boxes)), dtype=np.float32)
                for i, tb in enumerate(active_boxes):
                    for j, db in enumerate(det_boxes):
                        cost[i, j] = 1.0 - iou(tb, db)
                r_ind, c_ind = linear_sum_assignment(cost)
                matched_dets = set()
                for ri, cj in zip(r_ind, c_ind):
                    if 1.0 - cost[ri, cj] >= self.iou_thresh:
                        tid = active_ids[ri]
                        self.tracks[tid]["xyxy"] = det_boxes[cj]
                        self.tracks[tid]["age"] = 0
                        outputs.append(Track(track_id=tid, cls_name=cls_name, xyxy=det_boxes[cj]))
                        matched_dets.add(cj)
                # unmatched detections → new tracks
                for j, db in enumerate(det_boxes):
                    if j not in matched_dets:
                        tid = self.next_id
                        self.next_id += 1
                        self.tracks[tid] = {"cls_name": cls_name, "xyxy": db, "age": 0}
                        outputs.append(Track(track_id=tid, cls_name=cls_name, xyxy=db))
            else:
                # no active tracks of this class → new tracks for all dets
                for db in det_boxes:
                    tid = self.next_id
                    self.next_id += 1
                    self.tracks[tid] = {"cls_name": cls_name, "xyxy": db, "age": 0}
                    outputs.append(Track(track_id=tid, cls_name=cls_name, xyxy=db))
        # prune old tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]
        return outputs

class _ByteTrackWrapper:
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30):
        self.fallback = SimpleIOUTracker(iou_thresh=iou_thresh, max_age=max_age)
        self.use_fallback = True
        self.track_meta: Dict[int, Dict[str, Any]] = {}
        self.iou_match_thresh = max(0.05, min(0.7, iou_thresh))
        try:
            from supervision.tracker.byte_tracker import BYTETracker  # type: ignore
            try:
                from supervision.detection.core import Detections  # type: ignore
            except Exception:
                from supervision import Detections  # type: ignore
            self._Detections = Detections
            self.bt = BYTETracker()
            self.use_fallback = False
        except Exception:
            self.bt = None

    def update(self, detections) -> List[Track]:
        if self.use_fallback or self.bt is None:
            return self.fallback.update(detections)
        try:
            import numpy as np

            xyxy = np.array([d.xyxy for d in detections], dtype=np.float32) if detections else np.zeros((0, 4), np.float32)
            conf = np.array([getattr(d, "conf", 0.0) for d in detections], dtype=np.float32) if detections else np.zeros((0,), np.float32)
            dets = self._Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros((len(detections),), dtype=int) if detections else np.zeros((0,), dtype=int),
            )
            tracks = self.bt.update_with_detections(dets)

            outputs: List[Track] = []
            used_meta: Dict[int, Dict[str, Any]] = {}

            det_boxes = [tuple(d.xyxy) for d in detections]
            det_classes = [getattr(d, "cls_name", "object") for d in detections]

            for tr in tracks:
                tid = int(getattr(tr, "id", getattr(tr, "track_id", 0)))
                box = getattr(tr, "bbox", getattr(tr, "xyxy", None))
                if box is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                cls_name = self._match_class((x1, y1, x2, y2), det_boxes, det_classes, tid)
                used_meta[tid] = {"cls_name": cls_name}
                outputs.append(Track(track_id=tid, cls_name=cls_name, xyxy=(x1, y1, x2, y2)))

            self._prune_meta(used_meta)

            if not outputs:
                # Fall back to IoU tracker when ByteTrack yields nothing
                return self.fallback.update(detections)
            return outputs
        except Exception:
            self.use_fallback = True
            return self.fallback.update(detections)

    def _match_class(
        self,
        track_box: Tuple[float, float, float, float],
        det_boxes: List[Tuple[float, float, float, float]],
        det_classes: List[str],
        track_id: int,
    ) -> str:
        best_cls = None
        best_iou = 0.0
        for idx, box in enumerate(det_boxes):
            iou_val = iou(track_box, box)
            if iou_val > best_iou and iou_val >= self.iou_match_thresh:
                best_iou = iou_val
                best_cls = det_classes[idx]
        if best_cls is None:
            prev = self.track_meta.get(track_id)
            if prev:
                return prev.get("cls_name", "object")
            return "object"
        return best_cls

    def _prune_meta(self, fresh_meta: Dict[int, Dict[str, Any]]) -> None:
        self.track_meta = fresh_meta


class _OCSortWrapper:
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30):
        self.fallback = SimpleIOUTracker(iou_thresh=iou_thresh, max_age=max_age)
        self.use_fallback = True
        self.track_meta: Dict[int, Dict[str, Any]] = {}
        self.iou_match_thresh = max(0.05, min(0.7, iou_thresh))
        try:
            from ocsort.ocsort import OCSort  # type: ignore
            self.oc = OCSort()
            self.use_fallback = False
        except Exception:
            self.oc = None

    def update(self, detections) -> List[Track]:
        if self.use_fallback or self.oc is None:
            return self.fallback.update(detections)
        try:
            import numpy as np
            dets = []
            for d in detections:
                x1, y1, x2, y2 = d.xyxy
                dets.append([x1, y1, x2, y2, d.conf])
            dets = np.array(dets, dtype=np.float32) if dets else np.zeros((0, 5), np.float32)
            tracks = self.oc.update(dets)
            outputs: List[Track] = []
            det_boxes = [tuple(d.xyxy) for d in detections]
            det_classes = [getattr(d, "cls_name", "object") for d in detections]
            used_meta: Dict[int, Dict[str, Any]] = {}
            for t in tracks:
                x1, y1, x2, y2, tid = t[:5]
                bbox = (float(x1), float(y1), float(x2), float(y2))
                cls_name = self._match_class(bbox, det_boxes, det_classes, int(tid))
                used_meta[int(tid)] = {"cls_name": cls_name}
                outputs.append(Track(track_id=int(tid), cls_name=cls_name, xyxy=bbox))
            self._prune_meta(used_meta)
            if not outputs:
                return self.fallback.update(detections)
            return outputs
        except Exception:
            self.use_fallback = True
            return self.fallback.update(detections)

    def _match_class(
        self,
        track_box: Tuple[float, float, float, float],
        det_boxes: List[Tuple[float, float, float, float]],
        det_classes: List[str],
        track_id: int,
    ) -> str:
        best_cls = None
        best_iou = 0.0
        for idx, box in enumerate(det_boxes):
            iou_val = iou(track_box, box)
            if iou_val > best_iou and iou_val >= self.iou_match_thresh:
                best_iou = iou_val
                best_cls = det_classes[idx]
        if best_cls is None:
            prev = self.track_meta.get(track_id)
            if prev:
                return prev.get("cls_name", "object")
            return "object"
        return best_cls

    def _prune_meta(self, fresh_meta: Dict[int, Dict[str, Any]]) -> None:
        self.track_meta = fresh_meta


def build_tracker(backend: str, **kwargs: Any):
    backend = (backend or "iou").lower()
    if backend == "bytetrack":
        return _ByteTrackWrapper()
    if backend == "ocsort":
        return _OCSortWrapper()
    return SimpleIOUTracker(iou_thresh=0.3, max_age=kwargs.get("max_age", 30))

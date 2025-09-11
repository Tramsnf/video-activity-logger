from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
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

def build_tracker(backend: str):
    return SimpleIOUTracker(iou_thresh=0.3, max_age=30)
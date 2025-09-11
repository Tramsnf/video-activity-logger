from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class Detection:
    xyxy: Tuple[float,float,float,float]
    conf: float
    cls_name: str

class MockDetector:
    def __init__(self, classes: List[str]):
        self.classes = classes
    def __call__(self, frame) -> List[Detection]:
        # Returns empty detections; used to test pipeline wiring
        return []

class YoloDetector:
    def __init__(self, model_path: str, classes: List[str], min_conf: float=0.4,
                 min_box_area: int=900, roi_mask=None):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.classes = set(classes)
        self.names = self.model.model.names
        self.min_conf = min_conf
        self.min_box_area = min_box_area
        self.roi = roi_mask  # np.uint8 mask or None

    def __call__(self, frame) -> List[Detection]:
        res = self.model(frame, imgsz=960, conf=self.min_conf, verbose=False)[0]
        dets = []
        if res.boxes is None: return dets
        for b, c, k in zip(res.boxes.xyxy.cpu().numpy(),
                           res.boxes.conf.cpu().numpy(),
                           res.boxes.cls.cpu().numpy().astype(int)):
            name = self.names[int(k)]
            if name not in self.classes: continue
            x1,y1,x2,y2 = b.tolist()
            if (x2-x1)*(y2-y1) < self.min_box_area: continue
            if self.roi is not None:
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                if self.roi[cy, cx] == 0:   # outside ROI
                    continue
            dets.append(Detection((x1,y1,x2,y2), float(c), name))
        return dets

def build_detector(backend: str, model_path: Optional[str], classes: List[str]):
    if backend == "yolo":
        if not model_path:
            raise ValueError("YOLO backend requires model_path")
        return YoloDetector(model_path, classes)
    return MockDetector(classes)

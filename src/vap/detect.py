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
    def __init__(self, model_path: str, classes: List[str]):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.classes = set(classes)
        self.names = self.model.model.names

    def __call__(self, frame) -> List[Detection]:
        res = self.model(frame, imgsz=640, verbose=False)[0]
        dets = []
        if res.boxes is None: 
            return dets
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clses = res.boxes.cls.cpu().numpy().astype(int)
        for b, c, k in zip(boxes, confs, clses):
            name = self.names[int(k)]
            if name in self.classes:
                dets.append(Detection(tuple(b.tolist()), float(c), name))
        return dets

def build_detector(backend: str, model_path: Optional[str], classes: List[str]):
    if backend == "yolo":
        if not model_path:
            raise ValueError("YOLO backend requires model_path")
        return YoloDetector(model_path, classes)
    return MockDetector(classes)

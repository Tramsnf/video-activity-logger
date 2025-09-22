from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence
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
    def infer(self, frames: Sequence[np.ndarray]) -> List[List[Detection]]:
        return [[] for _ in frames]

class YoloDetector:
    def __init__(
        self,
        model_path: str,
        classes: List[str],
        min_conf: float = 0.4,
        min_box_area: int = 900,
        roi_mask=None,
        imgsz: int = 960,
        device: Optional[str] = None,
        fp16: bool = False,
        max_det: int = 200,
    ):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.classes = set(classes)
        # names attr can be at model.names (v8) or model.model.names (older)
        self.names = getattr(self.model, "names", None) or getattr(self.model, "model", None).names
        self.min_conf = float(min_conf)
        self.min_box_area = float(min_box_area)
        self.roi = roi_mask  # np.uint8 mask or None
        self.imgsz = int(imgsz)
        self.device = device
        self.fp16 = bool(fp16)
        self.max_det = int(max(1, max_det))

    def _filter_one(self, boxes, confs, clses) -> List[Detection]:
        dets: List[Detection] = []
        if boxes is None:
            return dets
        for b, c, k in zip(boxes, confs, clses):
            name = self.names[int(k)]
            if name not in self.classes:
                continue
            x1, y1, x2, y2 = b.tolist()
            conf = float(c)
            if conf < self.min_conf:
                continue
            if (x2 - x1) * (y2 - y1) < self.min_box_area:
                continue
            if self.roi is not None:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if cy < 0 or cx < 0 or cy >= self.roi.shape[0] or cx >= self.roi.shape[1] or self.roi[cy, cx] == 0:
                    continue
            dets.append(Detection((float(x1), float(y1), float(x2), float(y2)), conf, name))
        return dets

    def infer(self, frames: Sequence[np.ndarray]) -> List[List[Detection]]:
        if len(frames) == 0:
            return []
        # ultralytics model can accept list of images
        res_list = self.model(
            list(frames),
            imgsz=self.imgsz,
            conf=self.min_conf,
            device=self.device,
            half=self.fp16,
            verbose=False,
            max_det=self.max_det,
        )
        outputs: List[List[Detection]] = []
        for res in res_list:
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else None
            confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else None
            clses = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else None
            outputs.append(self._filter_one(boxes, confs, clses) if boxes is not None else [])
        return outputs

    def __call__(self, frame) -> List[Detection]:
        return self.infer([frame])[0]

def build_detector(backend: str, model_path: Optional[str], classes: List[str], **kwargs):
    if backend == "yolo":
        if not model_path:
            raise ValueError("YOLO backend requires model_path")
        return YoloDetector(model_path, classes, **kwargs)
    return MockDetector(classes)

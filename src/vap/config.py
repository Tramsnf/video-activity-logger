from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import yaml, pathlib

class DetectConfig(BaseModel):
    backend: str = Field(default="mock", description="mock | yolo")
    model_path: Optional[str] = None
    classes: List[str] = Field(default_factory=lambda: ["person","truck","pallet"])
    min_conf: float = 0.4
    min_box_area: int = 900
    imgsz: int = 960
    batch_size: int = 1
    stride: int = Field(default=1, description="Run detector every N frames (1 = every frame)")
    prefetch_frames: int = Field(default=0, description="Number of frames to prefetch asynchronously for the detector")
    device: Optional[str] = Field(default="auto", description="'auto', 'cpu', 'mps', CUDA id, or CSV list")
    max_det: int = Field(default=1000, description="Max detections per frame (helps keep NMS fast)")
    fp16: bool = False
    iou: float = Field(default=0.45, description="NMS IoU threshold")
    agnostic_nms: bool = Field(default=False, description="Run NMS without class separation")

class TrackConfig(BaseModel):
    backend: str = Field(default="bytetrack")
    max_age: int = 30

class Thresholds(BaseModel):
    pixels_per_meter: float = 90.0
    fps_target: int = 30
    # truck thresholds (m/s)
    fl_speed_drive: float = 0.20
    fl_speed_stop:  float = 0.05
    # Human thresholds (m/s)
    hu_speed_walk:  float = 0.50
    hu_speed_wait:  float = 0.08
    # Debounce / durations
    debounce_frames: int = 6
    min_state_dur_s: float = 1.0   # don’t emit states shorter than this
    merge_gap_s:     float = 0.5   # merge states separated by tiny gaps
    # Action heuristics
    action_attach_dist_m: float = 1.0
    action_detach_dist_m: float = 1.5
    action_attach_frames: int = 5
    action_detach_frames: int = 5

class PipelineConfig(BaseModel):
    video_id: str = "video_demo"
    source_camera: str = "cam1"
    roi_mask_path: Optional[str] = None
    zones_path: Optional[str] = None
    detect: DetectConfig = DetectConfig()
    track: TrackConfig = TrackConfig()
    thresholds: Thresholds = Thresholds()
    taxonomy_path: str = "configs/taxonomy.yaml"
    mapping_path: str = "configs/mapping.csv"

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_config(path: str) -> PipelineConfig:
    data = load_yaml(path)
    return PipelineConfig(**data)

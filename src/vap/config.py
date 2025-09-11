from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import yaml, pathlib

class DetectConfig(BaseModel):
    backend: str = Field(default="mock", description="mock | yolo")
    model_path: Optional[str] = None
    classes: List[str] = Field(default_factory=lambda: ["person","forklift","pallet"])

class TrackConfig(BaseModel):
    backend: str = Field(default="bytetrack")
    max_age: int = 30

class Thresholds(BaseModel):
    pixels_per_meter: float = 90.0
    fps_target: int = 30
    # Speed (m/s)
    speed_drive: float = 0.2
    speed_stop: float = 0.05
    # Debounce (frames)
    debounce_frames: int = 6  # ~0.2s at 30fps

class PipelineConfig(BaseModel):
    video_id: str = "video_demo"
    source_camera: str = "cam1"
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

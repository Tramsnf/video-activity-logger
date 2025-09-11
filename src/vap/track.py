from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class Track:
    track_id: int
    cls_name: str
    xyxy: Tuple[float,float,float,float]

class MockTracker:
    def __init__(self):
        self.next_id = 1
    def update(self, detections) -> List[Track]:
        # Assign a temporary unique id per detection (no temporal coherence)
        tracks = []
        for d in detections:
            tracks.append(Track(self.next_id, d.cls_name, d.xyxy))
            self.next_id += 1
        return tracks

def build_tracker(backend: str):
    # You can later wire ByteTrack/DeepSORT here
    return MockTracker()

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import yaml


def point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    # Ray casting algorithm for point in polygon
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def centroid(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ( (x1 + x2) / 2.0, (y1 + y2) / 2.0 )


@dataclass
class Zones:
    zones: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

    @staticmethod
    def load(path: str) -> "Zones":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        zones = {}
        for z in data.get("zones", []):
            name = z.get("name")
            poly = z.get("polygon") or z.get("points") or []
            if name and poly:
                zones[name] = [tuple(map(float, p)) for p in poly]
        return Zones(zones=zones)

    def actors_in(self, xyxy: Tuple[float, float, float, float]) -> List[str]:
        cx, cy = centroid(xyxy)
        hits = []
        for name, poly in self.zones.items():
            if point_in_poly(cx, cy, poly):
                hits.append(name)
        return hits


class ZoneEventer:
    def __init__(self, zones: Zones):
        self.zones = zones
        self.prev_in: Dict[str, set] = {}  # actor_id -> set(zone_names)

    def update(self, tracks, frame_idx: int, fps: float, source_camera: str, video_id: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for trk in tracks:
            aid = f"{trk.cls_name}_{trk.track_id}"
            curr = set(self.zones.actors_in(trk.xyxy))
            prev = self.prev_in.get(aid, set())
            # enters
            for name in curr - prev:
                events.append({
                    "video_id": video_id,
                    "actor_id": aid,
                    "actor_type": trk.cls_name,
                    "activity": "ZONE_ENTER",
                    "start_time_s": frame_idx / fps,
                    "end_time_s": frame_idx / fps,
                    "duration_s": 0.0,
                    "confidence": 0.9,
                    "source_camera": source_camera,
                    "attributes": {"zone": name},
                })
            # exits
            for name in prev - curr:
                events.append({
                    "video_id": video_id,
                    "actor_id": aid,
                    "actor_type": trk.cls_name,
                    "activity": "ZONE_EXIT",
                    "start_time_s": frame_idx / fps,
                    "end_time_s": frame_idx / fps,
                    "duration_s": 0.0,
                    "confidence": 0.9,
                    "source_camera": source_camera,
                    "attributes": {"zone": name},
                })
            self.prev_in[aid] = curr
        return events


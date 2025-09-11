from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any
import math, numpy as np

@dataclass
class ActorState:
    actor_id: str
    actor_type: str
    state: str | None = None
    state_start_f: int = 0
    last_pos: Tuple[float,float] | None = None
    speed_hist: List[float] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)

def centroid(xyxy):
    x1,y1,x2,y2 = xyxy
    return ( (x1+x2)/2.0, (y1+y2)/2.0 )

def smooth_speed(hist: List[float], k: int = 5) -> float:
    if not hist: return 0.0
    if len(hist) < k: return float(np.mean(hist))
    return float(np.mean(hist[-k:]))

def update_state_machine(actors: Dict[str, ActorState], tracks, frame_idx: int, fps: float,
                         pixels_per_meter: float, speed_drive: float, speed_stop: float, debounce_frames: int,
                         source_camera: str, video_id: str) -> List[Dict[str, Any]]:
    events = []
    for trk in tracks:
        tid = f"{trk.cls_name}_{trk.track_id}"
        cx, cy = centroid(trk.xyxy)
        a = actors.get(tid, ActorState(actor_id=tid, actor_type="forklift" if trk.cls_name=="forklift" else "human", state=None, state_start_f=frame_idx))

        # speed calc
        speed_mps = 0.0
        if a.last_pos is not None:
            dx = (cx - a.last_pos[0]) / pixels_per_meter
            dy = (cy - a.last_pos[1]) / pixels_per_meter
            speed_mps = math.hypot(dx, dy) * fps
        a.speed_hist.append(speed_mps)
        a.last_pos = (cx, cy)

        s = smooth_speed(a.speed_hist)
        if a.actor_type == "forklift":
            drive = s > speed_drive
            wait  = s <= speed_stop
            target = "DRIVE" if drive else ("WAIT" if wait else (a.state or "WAIT"))
        else:
            walk = s > speed_drive  # Assuming speed_drive is used for humans as well; adjust if needed
            wait = s <= speed_stop  # Assuming speed_stop is used for humans as well; adjust if needed
            target = "WALK" if walk else ("WAIT" if wait else (a.state or "WAIT"))

        # debounced transitions
        a.pending.append(target)
        if len(a.pending) >= debounce_frames and all(x == target for x in a.pending[-debounce_frames:]):
            if a.state != target:
                if a.state is not None:
                    events.append({
                        "video_id": video_id,
                        "actor_id": a.actor_id,
                        "actor_type": a.actor_type,
                        "activity": a.state,
                        "start_time_s": a.state_start_f / fps,
                        "end_time_s": frame_idx / fps,
                        "duration_s": (frame_idx - a.state_start_f) / fps,
                        "confidence": 0.85,
                        "source_camera": source_camera,
                        "attributes": {}
                    })
                a.state = target
                a.state_start_f = frame_idx

        actors[tid] = a
    return events

def close_open_states(actors: Dict[str, ActorState], last_frame: int, fps: float, source_camera: str, video_id: str):
    events = []
    for a in actors.values():
        if a.state is not None:
            events.append({
                "video_id": video_id,
                "actor_id": a.actor_id,
                "actor_type": a.actor_type,
                "activity": a.state,
                "start_time_s": a.state_start_f / fps,
                "end_time_s": last_frame / fps,
                "duration_s": (last_frame - a.state_start_f) / fps,
                "confidence": 0.7,
                "source_camera": source_camera,
                "attributes": {}
            })
    return events

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any
import math, numpy as np

VEHICLE_CLASSES = {"truck", "forklift"}
PERSON_CLASSES = {"person", "human", "worker", "driver"}


@dataclass
class ActorState:
    actor_id: str
    actor_type: str
    state: str | None = None
    state_start_f: int = 0
    last_pos: Tuple[float, float] | None = None
    speed_hist: List[float] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)
    occupant_id: str | None = None
    last_occupant_id: str | None = None
    occupant_missing: int = 0

def centroid(xyxy):
    x1,y1,x2,y2 = xyxy
    return ( (x1+x2)/2.0, (y1+y2)/2.0 )

def smooth_speed(hist: List[float], k: int = 5) -> float:
    if not hist: return 0.0
    if len(hist) < k: return float(np.mean(hist))
    return float(np.mean(hist[-k:]))


def actor_type_from_class(cls_name: str) -> str:
    name = (cls_name or "").lower()
    if name in VEHICLE_CLASSES:
        return name
    if name in PERSON_CLASSES:
        return "human"
    return name or "object"


def build_occupancy_maps(tracks, actors: Dict[str, ActorState]):
    occupants_by_vehicle: Dict[str, str] = {}
    vehicle_by_person: Dict[str, str] = {}

    vehicles = []
    persons = []
    for trk in tracks:
        cls_name = getattr(trk, "cls_name", "")
        tid = f"{cls_name}_{trk.track_id}"
        low = (cls_name or "").lower()
        if low in VEHICLE_CLASSES:
            vehicles.append((tid, trk))
        elif low in PERSON_CLASSES:
            persons.append((tid, trk))

    for vt_id, vtrk in vehicles:
        vx1, vy1, vx2, vy2 = vtrk.xyxy
        vcx, vcy = centroid(vtrk.xyxy)
        vw = max(1.0, float(vx2) - float(vx1))
        vh = max(1.0, float(vy2) - float(vy1))
        search_radius = max(20.0, max(vw, vh) * 0.8)
        prev_actor = actors.get(vt_id)
        preferred_id = None
        if prev_actor is not None:
            preferred_id = prev_actor.occupant_id or prev_actor.last_occupant_id
        best_tid = None
        best_score = (float("inf"), float("inf"))
        for pt_id, ptrk in persons:
            px, py = centroid(ptrk.xyxy)
            dist = math.hypot(vcx - px, vcy - py)
            inside = vx1 <= px <= vx2 and vy1 <= py <= vy2
            near = inside or dist <= search_radius
            if not near:
                continue
            priority = 1
            if inside:
                priority = 0
            if preferred_id and pt_id == preferred_id:
                priority = -1
            score = (priority, dist)
            if score < best_score:
                best_score = score
                best_tid = pt_id
        if best_tid is not None:
            occupants_by_vehicle[vt_id] = best_tid
            vehicle_by_person[best_tid] = vt_id

    return occupants_by_vehicle, vehicle_by_person

def update_state_machine(
    actors: Dict[str, ActorState],
    tracks,
    frame_idx: int,
    fps: float,
    pixels_per_meter: float,
    fl_speed_drive: float,
    fl_speed_stop: float,
    hu_speed_walk: float,
    hu_speed_wait: float,
    debounce_frames: int,
    source_camera: str,
    video_id: str,
) -> List[Dict[str, Any]]:
    events = []
    occupants_by_vehicle, vehicle_by_person = build_occupancy_maps(tracks, actors)
    for trk in tracks:
        tid = f"{trk.cls_name}_{trk.track_id}"
        actor_type = actor_type_from_class(getattr(trk, "cls_name", ""))
        if actor_type not in ("truck", "forklift", "human"):
            continue
        cx, cy = centroid(trk.xyxy)
        a = actors.get(tid)
        if a is None:
            a = ActorState(actor_id=tid, actor_type=actor_type, state=None, state_start_f=frame_idx)
        else:
            a.actor_type = actor_type

        # speed calc
        speed_mps = 0.0
        if a.last_pos is not None:
            dx = (cx - a.last_pos[0]) / pixels_per_meter
            dy = (cy - a.last_pos[1]) / pixels_per_meter
            speed_mps = math.hypot(dx, dy) * fps
        is_occupant_person = tid in vehicle_by_person
        if is_occupant_person:
            # Treat riders as stationary to avoid WALK states and skip event emission
            speed_mps = 0.0
        a.speed_hist.append(speed_mps)
        a.last_pos = (cx, cy)

        s = smooth_speed(a.speed_hist)
        # Vehicles (truck or forklift) use vehicle thresholds; humans use human thresholds
        occupant_id = occupants_by_vehicle.get(tid)
        a.occupant_id = occupant_id
        if a.occupant_id:
            a.last_occupant_id = a.occupant_id
            a.occupant_missing = 0
        else:
            a.occupant_missing = min(a.occupant_missing + 1, 60)
        if a.actor_type in ("truck", "forklift"):
            drive = s > fl_speed_drive
            wait  = s <= fl_speed_stop
            target = "DRIVE" if drive else ("WAIT" if wait else (a.state or "WAIT"))
        else:
            if is_occupant_person:
                a.state = None
                a.state_start_f = frame_idx
                a.pending.clear()
                actors[tid] = a
                continue
            walk = s > hu_speed_walk
            wait = s <= hu_speed_wait
            target = "WALK" if walk else ("WAIT" if wait else (a.state or "WAIT"))

        # debounced transitions
        a.pending.append(target)
        if len(a.pending) >= debounce_frames and all(x == target for x in a.pending[-debounce_frames:]):
            if a.state != target:
                if a.state is not None:
                    attrs: Dict[str, Any] = {}
                    driver_id = a.occupant_id or a.last_occupant_id
                    if driver_id:
                        attrs["occupant_id"] = driver_id
                        attrs["role"] = "driver"
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
                        "attributes": attrs
                    })
                a.state = target
                a.state_start_f = frame_idx

        actors[tid] = a
    return events

def close_open_states(actors: Dict[str, ActorState], last_frame: int, fps: float, source_camera: str, video_id: str):
    events = []
    for a in actors.values():
        if a.state is not None:
            attrs: Dict[str, Any] = {}
            driver_id = a.occupant_id or a.last_occupant_id
            if driver_id:
                attrs["occupant_id"] = driver_id
                attrs["role"] = "driver"
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
                "attributes": attrs
            })
    return events

def postprocess_state_events(
    events: List[Dict[str, Any]],
    taxonomy,
    min_state_dur_s: float,
    merge_gap_s: float,
) -> List[Dict[str, Any]]:
    """Merge tiny gaps of identical states and drop too-short state intervals.
    Non-state events are preserved.
    """
    if not events:
        return events

    # Group by actor and sort by start time
    by_actor: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        aid = e.get("actor_id", "")
        by_actor.setdefault(aid, []).append(e)
    for aid in by_actor:
        by_actor[aid].sort(key=lambda x: (x.get("start_time_s", 0.0), x.get("end_time_s", 0.0)))

    merged_total: List[Dict[str, Any]] = []
    state_names = getattr(taxonomy, "states", set())

    for aid, evs in by_actor.items():
        # First pass: merge consecutive same-activity with small gaps
        merged: List[Dict[str, Any]] = []
        for e in evs:
            if e.get("activity") in state_names and merged and merged[-1].get("activity") == e.get("activity"):
                gap = e.get("start_time_s", 0.0) - merged[-1].get("end_time_s", 0.0)
                if gap <= merge_gap_s:
                    # Extend previous
                    merged[-1]["end_time_s"] = max(merged[-1]["end_time_s"], e.get("end_time_s", merged[-1]["end_time_s"]))
                    merged[-1]["duration_s"] = merged[-1]["end_time_s"] - merged[-1]["start_time_s"]
                    continue
            merged.append(e)

        # Second pass: drop too-short states
        filtered: List[Dict[str, Any]] = []
        for e in merged:
            if e.get("activity") in state_names:
                dur = float(e.get("duration_s", 0.0))
                if dur < min_state_dur_s:
                    # Drop this short state
                    continue
            filtered.append(e)

        merged_total.extend(filtered)

    merged_total.sort(key=lambda x: (x.get("start_time_s", 0.0), x.get("end_time_s", 0.0)))
    return merged_total

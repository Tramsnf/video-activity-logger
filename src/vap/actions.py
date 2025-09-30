from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import math


def centroid(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ( (x1 + x2) / 2.0, (y1 + y2) / 2.0 )


@dataclass
class CarryState:
    carrying: bool = False
    near_count: int = 0
    far_count: int = 0
    pallet_id: Optional[str] = None
    candidate_id: Optional[str] = None


class ActionHeuristics:
    """Simple GRAB_SKID / PLACE_SKID heuristic based on forklift↔pallet proximity
    and forklift WAIT state around the moment of engagement/release.
    """

    def __init__(self):
        self.state: Dict[str, CarryState] = {}

    def update(
        self,
        actors: Dict[str, Any],
        tracks: List[Any],
        frame_idx: int,
        fps: float,
        pixels_per_meter: float,
        thresholds: Any,
        source_camera: str,
        video_id: str,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        # Build simple lookups
        vehicle_classes = {"truck", "forklift"}
        forklift = [t for t in tracks if getattr(t, "cls_name", "") in vehicle_classes]
        pallets = [t for t in tracks if getattr(t, "cls_name", "") == "pallet"]

        if not forklift:
            return events

        attach_px = float(getattr(thresholds, "action_attach_dist_m", 1.0)) * pixels_per_meter
        detach_px = float(getattr(thresholds, "action_detach_dist_m", 1.5)) * pixels_per_meter
        attach_frames = int(getattr(thresholds, "action_attach_frames", 5))
        detach_frames = int(getattr(thresholds, "action_detach_frames", 5))

        # For each truck, check nearest pallet and proximity streaks
        for fk in forklift:
            cls = getattr(fk, "cls_name", "forklift")
            aid = f"{cls}_{fk.track_id}"
            actor_type = cls if cls in vehicle_classes else "forklift"
            st = self.state.get(aid, CarryState())

            # Find nearest pallet
            fk_c = centroid(fk.xyxy)
            nearest_id: Optional[str] = None
            nearest_dist: float = float("inf")
            for p in pallets:
                p_c = centroid(p.xyxy)
                d = math.hypot(fk_c[0] - p_c[0], fk_c[1] - p_c[1])
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_id = f"pallet_{p.track_id}"

            # Determine near/far based on thresholds
            is_near = nearest_dist <= attach_px if nearest_id is not None else False

            # Actor state (from states.py) for context
            astate = actors.get(aid)
            curr_mode = getattr(astate, "state", None)
            driver_id = None
            if astate is not None:
                driver_id = getattr(astate, "occupant_id", None) or getattr(astate, "last_occupant_id", None)

            # Update near/far streaks
            if is_near:
                # Reset candidate if it changes
                if st.candidate_id != nearest_id:
                    st.candidate_id = nearest_id
                    st.near_count = 0
                st.near_count += 1
                st.far_count = 0
            else:
                st.near_count = 0
                st.far_count += 1

            # Attach (GRAB_SKID)
            if not st.carrying and is_near and st.near_count >= attach_frames and curr_mode == "WAIT":
                st.carrying = True
                st.pallet_id = st.candidate_id
                attrs = {"pallet_id": st.pallet_id, "distance_px": nearest_dist}
                if driver_id:
                    attrs["occupant_id"] = driver_id
                    attrs["role"] = "driver"
                events.append({
                    "video_id": video_id,
                    "actor_id": aid,
                    "actor_type": actor_type,
                    "activity": "GRAB_SKID",
                    "start_time_s": frame_idx / fps,
                    "end_time_s": frame_idx / fps,
                    "duration_s": 0.0,
                    "confidence": 0.6,
                    "source_camera": source_camera,
                    "attributes": attrs,
                })

            # Detach (PLACE_SKID)
            detach_ready = st.carrying and st.far_count >= detach_frames and curr_mode == "WAIT"
            if detach_ready:
                dist_clear = True
                if nearest_id is not None:
                    dist_clear = nearest_dist >= detach_px
                if dist_clear:
                    st.carrying = False
                    attrs = {"pallet_id": st.pallet_id}
                    if driver_id:
                        attrs["occupant_id"] = driver_id
                        attrs["role"] = "driver"
                    events.append({
                        "video_id": video_id,
                        "actor_id": aid,
                        "actor_type": actor_type,
                        "activity": "PLACE_SKID",
                        "start_time_s": frame_idx / fps,
                        "end_time_s": frame_idx / fps,
                        "duration_s": 0.0,
                        "confidence": 0.6,
                        "source_camera": source_camera,
                        "attributes": attrs,
                    })
                    st.pallet_id = None
                    st.candidate_id = None
                    st.far_count = 0
            if st.carrying and st.candidate_id is None:
                st.candidate_id = st.pallet_id

            self.state[aid] = st

        return events


def build_action_heuristics() -> ActionHeuristics:
    return ActionHeuristics()

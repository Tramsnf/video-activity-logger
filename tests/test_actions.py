from types import SimpleNamespace

from vap.actions import ActionHeuristics
from vap.states import ActorState
from vap.track import Track


def make_track(cls_name, track_id, xyxy):
    return Track(track_id=track_id, cls_name=cls_name, xyxy=xyxy)


def test_detach_requires_distance():
    heur = ActionHeuristics()
    thresholds = SimpleNamespace(
        action_attach_dist_m=1.0,
        action_detach_dist_m=2.0,
        action_attach_frames=2,
        action_detach_frames=2,
    )
    actors = {
        "forklift_1": ActorState(actor_id="forklift_1", actor_type="forklift", state="WAIT"),
    }
    # Close pallet to attach
    all_events = []
    for frame_idx in range(3):
        tracks = [
            make_track("forklift", 1, (0.0, 0.0, 2.0, 2.0)),
            make_track("pallet", 5, (1.0, 1.0, 2.0, 2.0)),
        ]
        events = heur.update(
            actors=actors,
            tracks=tracks,
            frame_idx=frame_idx,
            fps=30.0,
            pixels_per_meter=10.0,
            thresholds=thresholds,
            source_camera="cam",
            video_id="vid",
        )
        all_events.extend(events)
    grab_events = [e for e in all_events if e["activity"] == "GRAB_SKID"]
    assert grab_events, "Attach event should fire when pallet stays near"

    # Keep pallet near — should NOT detach yet
    all_events.clear()
    for frame_idx in range(3, 6):
        tracks = [
            make_track("forklift", 1, (0.0, 0.0, 2.0, 2.0)),
            make_track("pallet", 5, (3.0, 1.0, 4.0, 2.0)),
        ]
        events = heur.update(
            actors=actors,
            tracks=tracks,
            frame_idx=frame_idx,
            fps=30.0,
            pixels_per_meter=10.0,
            thresholds=thresholds,
            source_camera="cam",
            video_id="vid",
        )
        all_events.extend(events)
    assert not any(e["activity"] == "PLACE_SKID" for e in all_events), "Do not detach when pallet still close"

    # Move pallet far enough to trigger detach
    all_events.clear()
    for frame_idx in range(6, 9):
        tracks = [
            make_track("forklift", 1, (0.0, 0.0, 2.0, 2.0)),
            make_track("pallet", 5, (30.0, 1.0, 32.0, 2.0)),
        ]
        events = heur.update(
            actors=actors,
            tracks=tracks,
            frame_idx=frame_idx,
            fps=30.0,
            pixels_per_meter=10.0,
            thresholds=thresholds,
            source_camera="cam",
            video_id="vid",
        )
        all_events.extend(events)
    assert any(e["activity"] == "PLACE_SKID" for e in all_events), "Detach once pallet is beyond detach distance"

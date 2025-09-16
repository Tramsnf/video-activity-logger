from __future__ import annotations
import unittest
from typing import List

from vap.states import update_state_machine
from vap.track import Track


def gen_moving_tracks(cls_name: str, track_id: int, start_xyxy, dpx_per_frame: float, frames: int) -> List[Track]:
    x1, y1, x2, y2 = start_xyxy
    tracks: List[Track] = []
    for i in range(frames):
        dx = dpx_per_frame * i
        tracks.append(Track(track_id=track_id, cls_name=cls_name, xyxy=(x1 + dx, y1, x2 + dx, y2)))
    return tracks


class TestStates(unittest.TestCase):
    def run_sequence(self, cls_name: str, speed_mps: float, fps: float, ppm: float, debounce: int) -> str | None:
        # Given desired speed in m/s, compute pixels moved per frame
        dpx_per_frame = speed_mps * ppm / fps
        # Run a few extra frames beyond debounce to allow state to latch
        frames = debounce + 3
        seq = gen_moving_tracks(cls_name, 1, (0.0, 0.0, 10.0, 10.0), dpx_per_frame, frames)
        actors = {}
        state = None
        for i, trk in enumerate(seq):
            ev = update_state_machine(
                actors=actors,
                tracks=[trk],
                frame_idx=i,
                fps=fps,
                pixels_per_meter=ppm,
                fl_speed_drive=0.20,
                fl_speed_stop=0.05,
                hu_speed_walk=0.50,
                hu_speed_wait=0.08,
                debounce_frames=debounce,
                source_camera="cam",
                video_id="vid",
            )
            # ignore events; we want the current state latched after debounce
            _ = ev
            state = actors.get(f"{cls_name}_1").state if actors else None
        return state

    def test_truck_uses_vehicle_thresholds(self):
        # speed 0.3 m/s: vehicle should be DRIVE; human would be WAIT
        s = self.run_sequence(cls_name="truck", speed_mps=0.3, fps=30.0, ppm=90.0, debounce=3)
        self.assertEqual(s, "DRIVE")

    def test_forklift_behaves_like_vehicle(self):
        s = self.run_sequence(cls_name="forklift", speed_mps=0.3, fps=30.0, ppm=90.0, debounce=3)
        self.assertEqual(s, "DRIVE")

    def test_human_uses_human_thresholds(self):
        s = self.run_sequence(cls_name="person", speed_mps=0.3, fps=30.0, ppm=90.0, debounce=3)
        self.assertEqual(s, "WAIT")


if __name__ == "__main__":
    unittest.main()

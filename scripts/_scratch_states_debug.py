from vap.states import update_state_machine
from vap.track import Track

def run(cls_name: str):
    fps = 30.0
    ppm = 90.0
    speed_mps = 0.3
    dpx_per_frame = speed_mps * ppm / fps
    actors = {}
    for i in range(3):
        dx = dpx_per_frame * i
        trk = Track(track_id=1, cls_name=cls_name, xyxy=(0.0+dx, 0.0, 10.0+dx, 10.0))
        update_state_machine(
            actors=actors,
            tracks=[trk],
            frame_idx=i,
            fps=fps,
            pixels_per_meter=ppm,
            fl_speed_drive=0.20,
            fl_speed_stop=0.05,
            hu_speed_walk=0.50,
            hu_speed_wait=0.08,
            debounce_frames=3,
            source_camera="cam",
            video_id="vid",
        )
    print(cls_name, sorted(actors.keys()))
    k = f"{cls_name}_1"
    a = actors.get(k)
    print("state for", k, "=", getattr(a, "state", None))

for name in ("truck","forklift","person"):
    run(name)


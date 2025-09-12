from __future__ import annotations
from typing import Iterator, Tuple, Optional

def probe_fps(video_path: str) -> Optional[float]:
    # Try decord
    try:
        import decord  # type: ignore
        vr = decord.VideoReader(video_path)
        return float(vr.get_avg_fps())
    except Exception:
        pass
    # Try PyAV
    try:
        import av  # type: ignore
        cn = av.open(video_path)
        for s in cn.streams.video:
            if s.average_rate is not None:
                return float(s.average_rate)
        return None
    except Exception:
        pass
    # Fallback to OpenCV
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps and fps > 0 else None
    except Exception:
        return None

def frames(video_path: str) -> Iterator[Tuple[int, any]]:
    # Prefer decord for speed
    try:
        import decord  # type: ignore
        import numpy as np
        vr = decord.VideoReader(video_path)
        for i in range(len(vr)):
            frame = vr[i].asnumpy()[:, :, ::-1]  # to BGR for detector consistency
            yield i + 1, frame
        return
    except Exception:
        pass

    # Fallback to PyAV
    try:
        import av  # type: ignore
        import numpy as np
        cn = av.open(video_path)
        idx = 0
        for frame in cn.decode(video=0):
            img = frame.to_ndarray(format="bgr24")
            idx += 1
            yield idx, img
        return
    except Exception:
        pass

    # Fallback to OpenCV
    import cv2  # type: ignore
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        yield idx, frame
    cap.release()

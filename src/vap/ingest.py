from __future__ import annotations
import cv2
from typing import Iterator, Tuple

def frames(video_path: str) -> Iterator[Tuple[int, any]]:
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        yield idx, frame
    cap.release()

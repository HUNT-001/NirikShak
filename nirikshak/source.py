"""Frame sources: camera index, video file, image folder, or HTTP snapshot URL.

Android (Termux): run the "IP Webcam" app and use http://127.0.0.1:8080/shot.jpg.
Pi 5: camera index 0 (USB) or a libcamera/v4l2 device.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def frames(src: str, max_frames: int | None = None, fps: float | None = None) -> Iterator[tuple[str, np.ndarray]]:
    n, period = 0, (1.0 / fps if fps else 0.0)

    def done() -> bool:
        return max_frames is not None and n >= max_frames

    p = Path(src)
    if p.is_dir():
        for f in sorted(x for x in p.rglob("*") if x.suffix.lower() in IMG_EXT):
            if done():
                return
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is not None:
                n += 1
                yield str(f.relative_to(p)), img
        return
    if src.startswith(("http://", "https://")):
        while not done():
            t0 = time.time()
            with urllib.request.urlopen(src, timeout=5) as r:
                img = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                n += 1
                yield f"snap_{n:06d}", img
            time.sleep(max(0.0, period - (time.time() - t0)))
        return
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source {src!r}")
    try:
        while not done():
            ok, img = cap.read()
            if not ok:
                return
            n += 1
            yield f"frame_{n:06d}", img
    finally:
        cap.release()

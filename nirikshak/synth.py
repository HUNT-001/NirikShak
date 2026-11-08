"""Synthetic periodic parts (PCB-like pad arrays) with injected defects.

Used by tests and `nirikshak demo` so the pipeline runs with no dataset.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFECTS = ("scratch", "missing_pad", "contamination", "none")


def make_part(rng: np.random.Generator, size: int = 512, period: float = 26.0,
              angle: float = 3.0, defect: str = "none") -> np.ndarray:
    img = np.full((size, size), 70, np.uint8)
    pad = int(period * 0.45)
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    n = int(size / period) + 4
    for i in range(-2, n):
        for j in range(-2, n):
            x, y = i * period, j * period
            xr, yr = c * x - s * y + 8, s * x + c * y + 8
            cv2.rectangle(img, (int(xr), int(yr)), (int(xr) + pad, int(yr) + pad), 190, -1)
    if defect == "scratch":
        x0, y0 = rng.integers(60, size - 200, 2)
        L, th = rng.integers(80, 160), rng.uniform(0, np.pi)
        cv2.line(img, (int(x0), int(y0)), (int(x0 + L * np.cos(th)), int(y0 + L * np.sin(th))), 245, 2)
    elif defect == "missing_pad":
        i, j = rng.integers(4, n - 6, 2)
        x, y = i * period, j * period
        xr, yr = c * x - s * y + 8, s * x + c * y + 8
        cv2.rectangle(img, (int(xr) - 1, int(yr) - 1), (int(xr) + pad + 1, int(yr) + pad + 1), 70, -1)
    elif defect == "contamination":
        cx, cy = rng.integers(60, size - 60, 2)
        cv2.circle(img, (int(cx), int(cy)), int(rng.integers(5, 10)), 20, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32)
    img += rng.normal(0, 4, img.shape) + np.linspace(-15, 15, size)[None, :]  # noise + uneven light
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def write_set(out: Path, n: int = 40, seed: int = 0, defect_rate: float = 0.4) -> list[tuple[Path, str]]:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for k in range(n):
        d = str(rng.choice(DEFECTS[:3])) if rng.random() < defect_rate else "none"
        p = out / f"part_{k:04d}_{d}.png"
        cv2.imwrite(str(p), make_part(rng, angle=float(rng.uniform(-6, 6)), defect=d))
        items.append((p, d))
    return items

# Vendored from github.com/HUNT-001/DriftSense (driftsense/v2/lattice.py), same author, MIT.
"""S1 — lattice extraction in the Fourier domain.

Pure function. Windowed DFT of a frame, find the two dominant non-collinear
reciprocal-lattice peaks (excluding DC), and return the primitive reciprocal
vectors, the dominant real-space period, orientation and a confidence.

Hot-path ops are KleidiCV-listed only (createHanningWindow via mulSpectrums-free
elementwise multiply, dft, magnitude, log, minMaxLoc). Peak search and subpixel
refinement are vectorised / O(1) neighbourhood maths, never Python pixel loops.

NumPy-2 note: the collinearity test uses the explicit 2-D determinant
k1[0]*c[1] - k1[1]*c[0]; np.cross on 2-vectors is removed in NumPy 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_MIN_PERIOD_PX = 4.0  # ignore peaks finer than this (near-Nyquist noise)
_MIN_FFT = 1024  # pad to at least this for sub-0.5px period resolution
_SIN_TOL = 0.2  # |sin(angle)| below this => treat two vectors as collinear


@dataclass
class Lattice:
    k1: np.ndarray  # primitive reciprocal vector, cycles/px (x, y)
    k2: np.ndarray  # second, non-collinear reciprocal vector, cycles/px
    period_px: float  # dominant real-space period = 1/|k1|
    angle_deg: float  # orientation of k1, reduced to (-90, 90]
    confidence: float  # in-band power explained by the two peaks (+ conjugates)


def _reduce_angle(deg: float) -> float:
    """Reduce an axis orientation to (-90, 90]."""
    a = (deg + 90.0) % 180.0 - 90.0
    return 90.0 if a == -90.0 else a


def _det2(a: np.ndarray, c: np.ndarray) -> float:
    """Explicit 2-D determinant (NumPy-2 safe; never np.cross on 2-vectors)."""
    return float(a[0] * c[1] - a[1] * c[0])


def _power_spectrum(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Windowed, zero-padded log-magnitude spectrum, DC-centred. Returns (P, N)."""
    img = np.asarray(image, dtype=np.float32)
    img = img - float(img.mean())
    h, w = img.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    img = img * win
    n = cv2.getOptimalDFTSize(max(h, w, _MIN_FFT))
    pad = np.zeros((n, n), dtype=np.float32)
    pad[:h, :w] = img
    dft = cv2.dft(pad, flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
    logmag = cv2.log(mag + 1.0)
    # DC-centre (index roll only — not a pixel loop).
    return np.fft.fftshift(logmag), n


def _subpixel(P: np.ndarray, iy: int, ix: int) -> tuple[float, float]:
    """Quadratic peak interpolation about (iy, ix); returns (dy, dx) sub-offsets."""

    def delta(a: float, b: float, c: float) -> float:
        denom = a - 2.0 * b + c
        if abs(denom) < 1e-12:
            return 0.0
        d = 0.5 * (a - c) / denom
        return float(np.clip(d, -0.5, 0.5))

    h, w = P.shape
    dx = delta(P[iy, ix - 1], P[iy, ix], P[iy, ix + 1]) if 0 < ix < w - 1 else 0.0
    dy = delta(P[iy - 1, ix], P[iy, ix], P[iy + 1, ix]) if 0 < iy < h - 1 else 0.0
    return dy, dx


def detect_lattice(image: np.ndarray) -> Lattice:
    """Find the two dominant non-collinear reciprocal-lattice peaks."""
    P, n = _power_spectrum(image)
    cy, cx = n // 2, n // 2

    # In-band annulus: suppress DC (and periods coarser than half the frame) and
    # the near-Nyquist noise floor.
    yy, xx = np.ogrid[:n, :n]
    r = np.hypot(yy - cy, xx - cx)
    r_dc = max(3.0, n / (0.5 * min(image.shape)))  # 1/max_period in bins
    r_hi = n / _MIN_PERIOD_PX
    band = (r >= r_dc) & (r <= r_hi)
    work = np.where(band, P, 0.0).astype(np.float32)
    bg = float(np.median(P[band])) if band.any() else 0.0

    def strongest() -> tuple[int, int, float]:
        _, mx, _, (px, py) = cv2.minMaxLoc(work)
        return py, px, mx

    def suppress(py: int, px: int, rad: float) -> None:
        work[(yy - py) ** 2 + (xx - px) ** 2 <= rad * rad] = 0.0

    # Peak 1: global in-band maximum.
    iy1, ix1, m1 = strongest()
    dy1, dx1 = _subpixel(P, iy1, ix1)
    k1 = np.array([(ix1 + dx1 - cx) / n, (iy1 + dy1 - cy) / n], dtype=np.float64)
    suppress(iy1, ix1, max(3.0, r_dc))
    suppress(n - iy1, n - ix1, max(3.0, r_dc))  # conjugate -k1

    # Peak 2: strongest remaining peak that is non-collinear with k1.
    k2 = np.array([-k1[1], k1[0]], dtype=np.float64)  # fallback: ⟂ to k1
    m2 = 0.0
    for _ in range(32):
        iy2, ix2, m2 = strongest()
        if m2 <= 0.0:
            m2 = 0.0
            break
        c = np.array([(ix2 - cx) / n, (iy2 - cy) / n], dtype=np.float64)
        sin_sep = abs(_det2(k1, c)) / (np.linalg.norm(k1) * np.linalg.norm(c) + 1e-12)
        if sin_sep >= _SIN_TOL:
            dy2, dx2 = _subpixel(P, iy2, ix2)
            k2 = np.array([(ix2 + dx2 - cx) / n, (iy2 + dy2 - cy) / n], dtype=np.float64)
            break
        suppress(iy2, ix2, max(3.0, r_dc))
        suppress(n - iy2, n - ix2, max(3.0, r_dc))

    mag1 = float(np.linalg.norm(k1))
    period_px = 1.0 / mag1 if mag1 > 1e-9 else float("inf")
    angle_deg = _reduce_angle(float(np.degrees(np.arctan2(k1[1], k1[0]))))

    # Confidence: mean prominence of the two peaks over the in-band background.
    def _prom(m: float) -> float:
        return (m - bg) / (m + 1e-9) if m > 0.0 else 0.0

    confidence = float(np.clip(0.5 * (_prom(m1) + _prom(m2)), 0.0, 1.0))

    return Lattice(k1=k1, k2=k2, period_px=period_px, angle_deg=angle_deg, confidence=confidence)

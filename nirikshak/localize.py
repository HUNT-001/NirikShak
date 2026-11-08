"""Training-free defect localisation.

Periodic surfaces (PCB arrays, wafer dies, woven fabric, grids): the DriftSense
lattice stage (FFT) recovers the two real-space repeat vectors; every pixel is
compared with the median of its 8 lattice neighbours. A defect exists in one
copy only, so the median rejects it and the residual lights up. No golden image
and no defect samples are needed.

Stochastic textures (steel, leather, coatings): fall back to residual against a
large median-filtered background.

Residuals are converted to robust z-scores (median / MAD), so one threshold
works across products and lighting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .lattice import Lattice, detect_lattice

MIN_NEIGHBOURS = 4


@dataclass
class LocalizeConfig:
    work_px: int = 512            # longest side of the working image
    lattice_min_conf: float = 0.35
    z_thr: float = 6.0            # robust z for a pixel to count as anomalous
    min_area_px: int = 12         # at work resolution
    smooth_sigma: float = 1.5
    texture_kernel_frac: float = 0.08


@dataclass
class Region:
    bbox: tuple[int, int, int, int]   # x, y, w, h in ORIGINAL image pixels
    area_frac: float
    peak_z: float
    mean_z: float
    elongation: float
    label: str = "unclassified"
    label_prob: float = 0.0
    severity: str = "minor"

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class Localization:
    mode: str                       # "lattice" | "texture"
    score: float                    # frame anomaly score = max smoothed z
    regions: list[Region] = field(default_factory=list)
    period_px: float | None = None
    lattice_conf: float = 0.0
    zmap: np.ndarray | None = None  # work-resolution z map (for overlays)


def _gray_f32(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return g.astype(np.float32) / (255.0 if g.dtype == np.uint8 else 1.0)


def real_basis(lat: Lattice) -> tuple[np.ndarray, np.ndarray]:
    """Real-space primitive vectors a1, a2 with k_i . a_j = delta_ij."""
    A = np.linalg.inv(np.stack([lat.k1, lat.k2]))
    return A[:, 0], A[:, 1]


def _shift(img: np.ndarray, v: np.ndarray) -> np.ndarray:
    M = np.float32([[1, 0, -v[0]], [0, 1, -v[1]]])  # out(x) = img(x + v)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))


def _nanmedian0(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median over axis 0 ignoring NaN; ~5x faster than np.nanmedian here."""
    srt = np.sort(stack, axis=0)  # NaN sorts last
    lo = np.clip((valid - 1) // 2, 0, None)[None]
    hi = np.clip(valid // 2, 0, None)[None]
    m = 0.5 * (np.take_along_axis(srt, lo, 0) + np.take_along_axis(srt, hi, 0))[0]
    m[valid == 0] = np.nan
    return m


def lattice_residual(g: np.ndarray, lat: Lattice) -> np.ndarray:
    a1, a2 = real_basis(lat)
    # 5x5 lattice neighbourhood (24 copies): pixels near the frame edge still
    # get >= 4 valid neighbours, so the edge band is inspected too.
    vecs = [i * a1 + j * a2 for i in range(-2, 3) for j in range(-2, 3) if i or j]
    stack = np.stack([_shift(g, v) for v in vecs])
    valid = np.count_nonzero(~np.isnan(stack), axis=0)
    ref = _nanmedian0(stack, valid)
    ref[valid < MIN_NEIGHBOURS] = np.nan
    # How much the neighbours disagree among themselves (sub-pixel edge
    # interpolation, local texture). A defect must beat that, not just noise.
    spread = _nanmedian0(np.abs(stack - ref[None]), valid)
    ok = ~np.isnan(ref)
    floor = float(np.median(spread[ok])) + 1e-3 if ok.any() else 1e-3
    r = np.abs(g - ref) / (np.nan_to_num(spread, nan=np.inf) + floor)
    return np.where(ok, r, 0.0).astype(np.float32)  # no valid neighbour: no evidence


def texture_residual(g: np.ndarray, frac: float) -> np.ndarray:
    k = max(5, int(min(g.shape) * frac)) | 1
    u8 = np.clip(g * 255.0, 0, 255).astype(np.uint8)
    bg = cv2.medianBlur(u8, min(k, 255)).astype(np.float32) / 255.0
    return np.abs(g - bg)


def robust_z(r: np.ndarray) -> np.ndarray:
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med))) * 1.4826
    return (r - med) / max(mad, 1e-4)


def _usable(lat: Lattice, shape: tuple[int, int], cfg: LocalizeConfig) -> bool:
    if lat.confidence < cfg.lattice_min_conf or not np.isfinite(lat.period_px):
        return False
    a1, a2 = real_basis(lat)
    span = max(np.linalg.norm(a1), np.linalg.norm(a2))
    return 4.0 <= span <= min(shape) / 4.0  # need >= 2 repeats each side


def localize(img: np.ndarray, cfg: LocalizeConfig | None = None) -> Localization:
    cfg = cfg or LocalizeConfig()
    H, W = img.shape[:2]
    s = min(1.0, cfg.work_px / max(H, W))
    g = _gray_f32(cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else img)

    lat = detect_lattice(g)
    if _usable(lat, g.shape, cfg):
        mode, r = "lattice", lattice_residual(g, lat)
    else:
        mode, r = "texture", texture_residual(g, cfg.texture_kernel_frac)

    z = cv2.GaussianBlur(robust_z(r).astype(np.float32), (0, 0), cfg.smooth_sigma)
    mask = (z > cfg.z_thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    regions: list[Region] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < cfg.min_area_px:
            continue
        zi = z[lab == i]
        pts = np.column_stack(np.nonzero(lab == i)[::-1]).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        elong = max(rw, rh) / max(1.0, min(rw, rh))
        regions.append(Region(
            bbox=(int(x / s), int(y / s), max(1, int(w / s)), max(1, int(h / s))),
            area_frac=float(area) / float(g.size),
            peak_z=float(zi.max()), mean_z=float(zi.mean()), elongation=float(elong)))
    regions.sort(key=lambda r_: r_.peak_z, reverse=True)

    return Localization(mode=mode, score=float(z.max()), regions=regions,
                        period_px=float(lat.period_px / s) if mode == "lattice" else None,
                        lattice_conf=float(lat.confidence), zmap=z)

"""Calibrated PASS / REJECT / ESCALATE gate and severity grading.

Thresholds are fitted on a labelled validation set (`nirikshak calibrate`):
  t_pass   : below it a part passes; chosen so the defect escape rate meets the
             target. The Wilson 95% upper bound is reported, not hidden.
  t_reject : above it a part is rejected without a human; chosen so the false
             reject rate on good parts meets its target.
  between  : ESCALATE to the operator, the system says it is not sure.
Isotonic regression (PAV, from DriftSense) maps score -> P(defect) for display.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .localize import Region

PASS, REJECT, ESCALATE = "PASS", "REJECT", "ESCALATE"


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 1.0
    p, z2 = k / n, z * z
    c = p + z2 / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return float(min(1.0, (c + h) / (1 + z2 / n)))


def pav(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators isotonic fit (nondecreasing)."""
    lv: list[float] = []
    cnt: list[int] = []
    for v in np.asarray(y, float):
        lv.append(v)
        cnt.append(1)
        while len(lv) > 1 and lv[-2] > lv[-1]:
            n = cnt[-2] + cnt[-1]
            lv[-2:] = [(lv[-2] * cnt[-2] + lv[-1] * cnt[-1]) / n]
            cnt[-2:] = [n]
    return np.repeat(lv, cnt)


@dataclass
class Calibration:
    t_pass: float = 10.0
    t_reject: float = 14.0
    knots_x: list[float] = field(default_factory=lambda: [0.0, 10.0, 14.0, 60.0])
    knots_p: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.9, 1.0])
    report: dict = field(default_factory=lambda: {"source": "uncalibrated defaults"})

    def p_defect(self, score: float) -> float:
        return float(np.interp(score, self.knots_x, self.knots_p))

    def decide(self, score: float) -> str:
        if score < self.t_pass:
            return PASS
        return REJECT if score >= self.t_reject else ESCALATE

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | str | None) -> "Calibration":
        if path and Path(path).exists():
            return cls(**json.loads(Path(path).read_text()))
        return cls()


def fit(scores_good: list[float], scores_defect: list[float],
        escape_target: float = 0.02, false_reject_target: float = 0.01) -> Calibration:
    g, d = np.sort(np.asarray(scores_good, float)), np.sort(np.asarray(scores_defect, float))
    if g.size == 0 or d.size == 0:
        raise ValueError("need both good and defective samples")
    # d_k: largest threshold whose escape rate (defects scoring below it) <= target.
    k_esc = int(math.floor(escape_target * d.size))
    d_k = float(d[min(k_esc, d.size - 1)])
    # g_k: smallest threshold whose false-reject rate (good parts at/above it) <= target.
    k_fr = int(math.floor(false_reject_target * g.size))
    g_k = float(g[max(0, g.size - 1 - k_fr)]) + 1e-6
    # Overlap -> band where both targets cannot be met. Gap -> no calibration
    # evidence either way. Both cases ESCALATE rather than guess.
    t_pass, t_reject = min(d_k, g_k), max(d_k, g_k)

    x = np.concatenate([g, d])
    y = np.concatenate([np.zeros(g.size), np.ones(d.size)])
    o = np.argsort(x)
    xs, ps = x[o], pav(y[o])
    keep = np.r_[True, np.diff(ps) != 0] | np.r_[np.diff(ps) != 0, True]

    esc = int((d < t_pass).sum())
    fr = int((g >= t_reject).sum())
    band = int(((x >= t_pass) & (x < t_reject)).sum())
    rep = {
        "n_good": int(g.size), "n_defect": int(d.size),
        "escape_rate": esc / d.size, "escape_wilson95_upper": wilson_upper(esc, d.size),
        "false_reject_rate": fr / g.size, "false_reject_wilson95_upper": wilson_upper(fr, g.size),
        "escalate_fraction": band / x.size,
        "targets": {"escape": escape_target, "false_reject": false_reject_target},
    }
    return Calibration(t_pass=t_pass, t_reject=t_reject, knots_x=xs[keep].round(4).tolist(),
                       knots_p=ps[keep].round(4).tolist(), report=rep)


@dataclass
class SeverityConfig:
    critical_labels: tuple[str, ...] = ("crack", "scratch", "missing_pad", "short", "open")
    critical_area_frac: float = 0.02
    major_area_frac: float = 0.004
    major_peak_z: float = 25.0


def grade(r: Region, cfg: SeverityConfig) -> str:
    if r.label in cfg.critical_labels or r.area_frac >= cfg.critical_area_frac:
        return "critical"
    if r.area_frac >= cfg.major_area_frac or r.peak_z >= cfg.major_peak_z:
        return "major"
    return "minor"

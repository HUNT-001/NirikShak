"""Inspector: localize -> classify -> grade -> gate -> log -> (async) advise."""
from __future__ import annotations

import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .advisor import Advisor, LlamaRunner, RunnerConfig
from .classify import DefectClassifier
from .gate import ESCALATE, PASS, REJECT, Calibration, SeverityConfig, grade
from .localize import LocalizeConfig, Region, localize
from .store import Store

COLORS = {PASS: (60, 180, 60), REJECT: (40, 40, 220), ESCALATE: (0, 170, 255)}


@dataclass
class Config:
    localize: LocalizeConfig = field(default_factory=LocalizeConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    calibration: str = "calibration.json"
    classifier_onnx: str = ""
    classifier_labels: str = ""
    db: str = "inspections.db"
    trend_window: int = 50

    @classmethod
    def load(cls, path: str | None) -> "Config":
        c = cls()
        if not path:
            return c
        d = tomllib.loads(Path(path).read_text())
        for sect, obj in (("localize", c.localize), ("severity", c.severity), ("runner", c.runner)):
            for k, v in d.pop(sect, {}).items():
                setattr(obj, k, tuple(v) if isinstance(v, list) else v)
        for k, v in d.items():
            setattr(c, k, v)
        return c


@dataclass
class Result:
    frame: str
    decision: str
    score: float
    p_defect: float
    mode: str
    regions: list[Region]
    timings_ms: dict
    row_id: int = 0


class Inspector:
    def __init__(self, cfg: Config, store: Store | None = None, advise: bool = True):
        self.cfg = cfg
        self.calib = Calibration.load(cfg.calibration)
        self.clf = DefectClassifier(cfg.classifier_onnx or None, cfg.classifier_labels or None)
        self.store = store or Store(cfg.db)
        self.advisor = Advisor(LlamaRunner(cfg.runner)) if advise else None
        if self.advisor:
            self.advisor.start(self.store.set_advice)

    def inspect(self, img: np.ndarray, frame: str = "") -> Result:
        t0 = time.perf_counter()
        loc = localize(img, self.cfg.localize)
        t1 = time.perf_counter()
        decision = self.calib.decide(loc.score)
        regions = loc.regions if decision != PASS else []
        self.clf(img, regions)
        for r in regions:
            r.severity = grade(r, self.cfg.severity)
        t2 = time.perf_counter()
        p = self.calib.p_defect(loc.score)
        timings = {"localize": round((t1 - t0) * 1e3, 1), "classify": round((t2 - t1) * 1e3, 1),
                   "total": round((t2 - t0) * 1e3, 1)}
        res = Result(frame, decision, loc.score, p, loc.mode, regions, timings)
        res.row_id = self.store.add(frame=frame, decision=decision, score=round(loc.score, 3),
                                    p_defect=round(p, 4), mode=loc.mode,
                                    regions=[r.as_dict() for r in regions[:5]], latency_ms=timings["total"])
        if self.advisor and decision != PASS:
            top = regions[0] if regions else Region((0, 0, 0, 0), 0.0, loc.score, loc.score, 1.0)
            event = {"decision": decision, "label": top.label, "severity": top.severity,
                     "p_defect": round(p, 3), "area_frac": round(top.area_frac, 5),
                     "n_regions": len(regions), "mode": loc.mode}
            self.advisor.submit(res.row_id, event, self.store.trend(self.cfg.trend_window))
        return res

    def close(self) -> None:
        if self.advisor:
            self.advisor.drain()


def annotate(img: np.ndarray, res: Result) -> np.ndarray:
    out = img.copy()
    col = COLORS[res.decision]
    for r in res.regions[:5]:
        x, y, w, h = r.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        cv2.putText(out, f"{r.label} {r.severity}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(out, f"{res.decision}  p={res.p_defect:.2f}  score={res.score:.1f}  {res.timings_ms['total']:.0f}ms",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
    return out

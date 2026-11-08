"""SQLite inspection log (one file, safe across the capture and advisor threads)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL, frame TEXT, decision TEXT NOT NULL, score REAL, p_defect REAL,
  mode TEXT, top_label TEXT, top_severity TEXT, regions TEXT,
  latency_ms REAL, advice TEXT
);
CREATE INDEX IF NOT EXISTS ix_ts ON inspections(ts);
"""


class Store:
    def __init__(self, path: str | Path = "inspections.db"):
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def add(self, *, frame: str, decision: str, score: float, p_defect: float, mode: str,
            regions: list[dict], latency_ms: float) -> int:
        top = regions[0] if regions else {}
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO inspections(ts,frame,decision,score,p_defect,mode,top_label,top_severity,"
                "regions,latency_ms) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (time.time(), frame, decision, score, p_defect, mode, top.get("label"),
                 top.get("severity"), json.dumps(regions), latency_ms))
            self.db.commit()
            return int(cur.lastrowid)

    def set_advice(self, row_id: int, advice: dict) -> None:
        with self._lock:
            self.db.execute("UPDATE inspections SET advice=? WHERE id=?", (json.dumps(advice), row_id))
            self.db.commit()

    def recent(self, n: int = 50) -> list[dict]:
        with self._lock:
            rows = self.db.execute("SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["regions"] = json.loads(d["regions"] or "[]")
            d["advice"] = json.loads(d["advice"]) if d["advice"] else None
            out.append(d)
        return out

    def trend(self, window: int = 50) -> dict:
        """Defect statistics over the last `window` parts vs the window before."""
        with self._lock:
            rows = self.db.execute("SELECT decision, top_label FROM inspections ORDER BY id DESC LIMIT ?",
                                   (2 * window,)).fetchall()
        cur, prev = rows[:window], rows[window:]
        rate = lambda rs: (sum(r["decision"] != "PASS" for r in rs) / len(rs)) if rs else 0.0  # noqa: E731
        labels: dict[str, int] = {}
        for r in cur:
            if r["decision"] != "PASS" and r["top_label"]:
                labels[r["top_label"]] = labels.get(r["top_label"], 0) + 1
        return {"parts": len(cur), "defect_rate": round(rate(cur), 3),
                "prev_defect_rate": round(rate(prev), 3) if prev else None, "by_label": labels}

    def stats(self) -> dict:
        with self._lock:
            rows = self.db.execute("SELECT decision, COUNT(*) c, AVG(latency_ms) l FROM inspections "
                                   "GROUP BY decision").fetchall()
        return {r["decision"]: {"count": r["c"], "avg_latency_ms": round(r["l"] or 0, 1)} for r in rows}

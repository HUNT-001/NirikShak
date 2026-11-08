"""nirikshak CLI.

  nirikshak run --source SRC [--serve 8000] [--save-dir out/] [--max N]
  nirikshak calibrate --good DIR --defect DIR [--out calibration.json]
  nirikshak demo [--n 60]                 synthetic parts, no dataset needed
  nirikshak tune-llm [--mock] [--budget 12]
  nirikshak advise --event '{...}'        one advisory call (tests the Llama runner)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from .gate import fit
from .pipeline import Config, Inspector, annotate
from .source import IMG_EXT, frames


def _scores(cfg: Config, folder: str) -> list[float]:
    from .localize import localize
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)
    return [localize(cv2.imread(str(p)), cfg.localize).score for p in files]


def cmd_calibrate(a: argparse.Namespace, cfg: Config) -> None:
    good, bad = _scores(cfg, a.good), _scores(cfg, a.defect)
    cal = fit(good, bad, a.escape_target, a.false_reject_target)
    cal.save(a.out)
    print(json.dumps({"t_pass": round(cal.t_pass, 3), "t_reject": round(cal.t_reject, 3), **cal.report}, indent=2))


def cmd_run(a: argparse.Namespace, cfg: Config) -> dict:
    insp = Inspector(cfg, advise=not a.no_advise)
    live = None
    if a.serve:
        from .server import LiveState, serve
        live = LiveState()
        serve(insp.store, live, port=a.serve, trend_window=cfg.trend_window)
        print(f"dashboard: http://localhost:{a.serve}")
    save = Path(a.save_dir) if a.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    lat: list[float] = []
    t0 = time.perf_counter()
    for name, img in frames(a.source, a.max, a.fps):
        res = insp.inspect(img, name)
        counts[res.decision] = counts.get(res.decision, 0) + 1
        lat.append(res.timings_ms["total"])
        if live or save:
            vis = annotate(img, res)
            if live:
                live.set(cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
            if save and res.decision != "PASS":
                cv2.imwrite(str(save / (Path(name).stem + f"_{res.decision}.jpg")), vis)
        if not a.quiet:
            top = res.regions[0] if res.regions else None
            print(f"{name:40s} {res.decision:8s} score={res.score:6.1f} p={res.p_defect:.2f} "
                  f"{(top.label + '/' + top.severity) if top else '':24s} {res.timings_ms['total']:.0f}ms")
    insp.close()
    lat.sort()
    summary = {"frames": len(lat), "decisions": counts,
               "latency_ms_p50": lat[len(lat) // 2] if lat else None,
               "latency_ms_p95": lat[int(0.95 * (len(lat) - 1))] if lat else None,
               "fps": round(len(lat) / (time.perf_counter() - t0), 2) if lat else None}
    print(json.dumps(summary))
    if a.serve and a.hold:
        print("serving; Ctrl+C to stop")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return summary


def cmd_demo(a: argparse.Namespace, cfg: Config) -> None:
    from .synth import write_set
    root = Path(a.dir)
    cal_items = write_set(root / "calib", n=a.n, seed=1, defect_rate=0.5)
    (root / "calib_good").mkdir(exist_ok=True)
    (root / "calib_defect").mkdir(exist_ok=True)
    for p, d in cal_items:
        p.rename(root / ("calib_good" if d == "none" else "calib_defect") / p.name)
    (root / "calib").rmdir()
    cfg.calibration = str(root / "calibration.json")
    cmd_calibrate(argparse.Namespace(good=str(root / "calib_good"), defect=str(root / "calib_defect"),
                                     out=cfg.calibration, escape_target=0.02, false_reject_target=0.01), cfg)
    items = write_set(root / "line", n=a.n, seed=2, defect_rate=0.3)
    cfg.db = str(root / "demo.db")
    Path(cfg.db).unlink(missing_ok=True)
    s = cmd_run(argparse.Namespace(source=str(root / "line"), serve=a.serve, save_dir=str(root / "flagged"),
                                   max=None, fps=None, no_advise=False, quiet=True, hold=a.serve is not None), cfg)
    truth = {p.name: d for p, d in items}
    from .store import Store
    rows = Store(cfg.db).recent(10_000)
    esc = sum(1 for r in rows if r["decision"] == "PASS" and truth[r["frame"]] != "none")
    fr = sum(1 for r in rows if r["decision"] == "REJECT" and truth[r["frame"]] == "none")
    print(json.dumps({"escapes": esc, "false_rejects": fr, "defective": sum(d != "none" for d in truth.values()),
                      "good": sum(d == "none" for d in truth.values()), **s}))


def cmd_tune(a: argparse.Namespace, cfg: Config) -> None:
    from anvil.objective import Objective

    from .llm_tune import RealEvaluator, load_golden, mock_evaluator, search
    obj = Objective(accuracy_floor=a.quality_floor, w_latency=1.0, w_memory=0.5, size_budget_mb=a.size_budget_mb)
    ev = mock_evaluator if a.mock else RealEvaluator(load_golden(a.golden), cfg.runner, Path(a.out_dir))
    r = search(ev, obj, budget=a.budget)
    Path(a.report).write_text(json.dumps(r, indent=2))
    print(json.dumps({k: r[k] for k in ("best", "best_metrics", "feasible", "evaluations", "grid_size")}, indent=2))


def cmd_advise(a: argparse.Namespace, cfg: Config) -> None:
    from .advisor import Advisor, LlamaRunner
    trend = json.loads(a.trend) if a.trend else {"parts": 50, "defect_rate": 0.04, "prev_defect_rate": 0.04, "by_label": {}}
    print(json.dumps(Advisor(LlamaRunner(cfg.runner)).advise(json.loads(a.event), trend), indent=2))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="nirikshak")
    ap.add_argument("--config", default=None, help="TOML config (see configs/default.toml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--source", required=True)
    r.add_argument("--serve", type=int, default=None)
    r.add_argument("--hold", action="store_true", help="keep dashboard up after the source ends")
    r.add_argument("--save-dir")
    r.add_argument("--max", type=int)
    r.add_argument("--fps", type=float)
    r.add_argument("--no-advise", action="store_true")
    r.add_argument("--quiet", action="store_true")

    c = sub.add_parser("calibrate")
    c.add_argument("--good", required=True)
    c.add_argument("--defect", required=True)
    c.add_argument("--out", default="calibration.json")
    c.add_argument("--escape-target", type=float, default=0.02)
    c.add_argument("--false-reject-target", type=float, default=0.01)

    d = sub.add_parser("demo")
    d.add_argument("--n", type=int, default=60)
    d.add_argument("--dir", default="demo_out")
    d.add_argument("--serve", type=int, default=None)

    t = sub.add_parser("tune-llm")
    t.add_argument("--mock", action="store_true")
    t.add_argument("--budget", type=int, default=12)
    t.add_argument("--quality-floor", type=float, default=0.9)
    t.add_argument("--size-budget-mb", type=float, default=1200.0)
    t.add_argument("--golden", default="data/golden_advice.jsonl")
    t.add_argument("--out-dir", default="models/llm")
    t.add_argument("--report", default="tune_report.json")

    v = sub.add_parser("advise")
    v.add_argument("--event", required=True)
    v.add_argument("--trend")

    a = ap.parse_args(argv)
    cfg = Config.load(a.config)
    {"run": cmd_run, "calibrate": cmd_calibrate, "demo": cmd_demo,
     "tune-llm": cmd_tune, "advise": cmd_advise}[a.cmd](a, cfg)


if __name__ == "__main__":
    sys.exit(main())

import json
import os
import stat
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from anvil.objective import Objective
from nirikshak.advisor import Advisor, LlamaRunner, RunnerConfig, extract_json, parse_observer, rule_advice
from nirikshak.gate import ESCALATE, PASS, REJECT, fit
from nirikshak.llm_tune import grid_size, mock_evaluator, search
from nirikshak.localize import localize
from nirikshak.pipeline import Config, Inspector
from nirikshak.store import Store
from nirikshak.synth import make_part

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("angle", [0.0, 4.0])
@pytest.mark.parametrize("defect", ["scratch", "missing_pad", "contamination"])
def test_lattice_finds_defect(defect, angle):
    loc = localize(make_part(np.random.default_rng(0), angle=angle, defect=defect))
    assert loc.mode == "lattice" and abs(loc.period_px - 26.0) < 0.5
    assert loc.score > 15 and loc.regions


def test_clean_part_scores_low():
    rng = np.random.default_rng(1)
    scores = [localize(make_part(rng, angle=a)).score for a in (-5, 0, 3, 6)]
    assert max(scores) < 12


def test_texture_fallback_on_non_periodic():
    rng = np.random.default_rng(2)
    img = (rng.normal(120, 6, (256, 256))).clip(0, 255).astype(np.uint8)
    img[100:104, 40:200] = 250
    loc = localize(np.dstack([img] * 3))
    assert loc.mode == "texture" and loc.regions


def test_calibration_gate():
    cal = fit(list(np.linspace(1, 10, 100)), list(np.linspace(8, 40, 100)), 0.02, 0.01)
    assert cal.t_pass <= cal.t_reject
    assert cal.decide(2) == PASS and cal.decide(39) == REJECT and cal.decide(9) == ESCALATE
    assert cal.report["escape_rate"] <= 0.02 and cal.report["false_reject_rate"] <= 0.01
    assert 0 <= cal.p_defect(9) <= cal.p_defect(30) <= 1


def test_golden_rules_consistent():
    for line in (ROOT / "data/golden_advice.jsonl").read_text().splitlines():
        c = json.loads(line)
        assert rule_advice(c["event"], c["trend"])["action"] == c["expected_action"]


def test_parse_llm_output():
    text = ('<|start_header_id|>assistant<|end_header_id|>\n\nSure! {"action": "rework", "root_cause": "x", '
            '"operator_check": "y", "urgency": "medium"}\nPyTorchObserver {"prompt_tokens":100,"generated_tokens":41,'
            '"model_load_start_ms":0,"model_load_end_ms":900,"inference_start_ms":1000,"prompt_eval_end_ms":1800,'
            '"first_token_ms":1810,"inference_end_ms":5810,"SCALING_FACTOR_UNITS_PER_SECOND":1000}')
    assert extract_json(text)["action"] == "REWORK"
    st = parse_observer(text)
    assert st["ttft_ms"] == 810 and abs(st["decode_tok_s"] - 40 / 4.01) < 1e-6


def _fake_runner(tmp: Path) -> Path:
    """Stand-in for ExecuTorch llama_main: same CLI flags, same stdout shape."""
    p = tmp / "llama_main"
    p.write_text(f"""#!{sys.executable}
import json, sys
prompt = [a for a in sys.argv if a.startswith('--prompt=')][0]
act = 'STOP_LINE' if '"critical"' in prompt and '0.2' in prompt else 'REWORK'
print(prompt[9:] + json.dumps({{"action": act, "root_cause": "fake", "operator_check": "fake", "urgency": "high"}}))
print('PyTorchObserver ' + json.dumps({{"prompt_tokens": 120, "generated_tokens": 30, "inference_start_ms": 0,
      "prompt_eval_end_ms": 600, "first_token_ms": 610, "inference_end_ms": 3510}}))
""")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_advisor_local_runner_and_fallback(tmp_path):
    rc = RunnerConfig(mode="local", binary=str(_fake_runner(tmp_path)))
    ev = {"decision": "REJECT", "label": "scratch", "severity": "major"}
    tr = {"parts": 50, "defect_rate": 0.04, "prev_defect_rate": 0.04, "by_label": {}}
    a = Advisor(LlamaRunner(rc)).advise(ev, tr)
    assert a["source"] == "llama-executorch" and a["action"] == "REWORK" and a["stats"]["ttft_ms"] == 610
    b = Advisor(LlamaRunner(RunnerConfig(mode="local", binary=str(tmp_path / "missing")))).advise(ev, tr)
    assert b["source"] == "rules" and b["fallback_reason"].startswith("runner error")


def test_pipeline_logs_and_advises(tmp_path):
    cfg = Config()
    cfg.db, cfg.calibration = str(tmp_path / "t.db"), str(tmp_path / "none.json")
    cfg.runner = RunnerConfig(mode="local", binary=str(_fake_runner(tmp_path)))
    insp = Inspector(cfg)
    rng = np.random.default_rng(4)
    r1 = insp.inspect(make_part(rng, defect="scratch"), "a")
    r2 = insp.inspect(make_part(rng), "b")
    insp.close()
    assert r1.decision == REJECT and r2.decision == PASS
    rows = {r["frame"]: r for r in Store(cfg.db).recent()}
    assert rows["a"]["advice"]["source"] == "llama-executorch" and rows["b"]["advice"] is None


def test_dashboard_serves(tmp_path):
    from nirikshak.server import LiveState, serve
    st = Store(tmp_path / "d.db")
    st.add(frame="x", decision="REJECT", score=30.0, p_defect=0.99, mode="lattice", regions=[], latency_ms=5.0)
    live = LiveState()
    live.set(b"\xff\xd8jpeg")
    httpd = serve(st, live, host="127.0.0.1", port=0)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    assert b"Nirikshak" in urllib.request.urlopen(base + "/").read()
    assert json.loads(urllib.request.urlopen(base + "/api/stats").read())["decisions"]["REJECT"]["count"] == 1
    assert urllib.request.urlopen(base + "/frame.jpg").read().startswith(b"\xff\xd8")
    httpd.shutdown()


def test_llm_tune_beats_grid():
    r = search(mock_evaluator, Objective(accuracy_floor=0.9, w_memory=0.5, size_budget_mb=1200), budget=14)
    assert r["feasible"] and r["evaluations"] < grid_size() / 4
    assert r["best_metrics"]["accuracy"] >= 0.9 and r["best_metrics"]["memory_mb"] <= 1200

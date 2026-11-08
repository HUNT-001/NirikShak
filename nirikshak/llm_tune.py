"""Anvil for ExecuTorch: pick the Llama export config for THIS device.

Reuses Anvil's constrained objective (anvil/objective.py): maximise speed and
size reduction subject to a quality floor and a size budget. The knobs are
ExecuTorch's XNNPACK export options instead of GGUF per-layer types:

  group_size        : 8da4w linear-weight quant group (32 | 64 | 128 | 256)
  embedding_quantize: None | "8,0" | "4,32"
  max_seq_length    : 384 | 768
  threads           : runner --cpu_threads (runtime only, no re-export)

Quality is task accuracy on data/golden_advice.jsonl (valid JSON + correct
action), not wikitext perplexity: it measures the job the model does here.

Search = Anvil-style greedy coordinate ascent with caching. Each evaluation
costs an export (~minutes) plus on-device runs, so the point is to find a good
config in far fewer evaluations than the full grid.
"""
from __future__ import annotations

import itertools
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from anvil.model import Metrics
from anvil.objective import Objective

from .advisor import SYSTEM, LlamaRunner, RunnerConfig, build_event_text, chat_prompt, extract_json

SPACE = {
    "group_size": (32, 64, 128, 256),
    "embedding_quantize": (None, "8,0", "4,32"),
    "max_seq_length": (384, 768),
    "threads": (2, 4, 6),
}


@dataclass(frozen=True)
class ETConfig:
    group_size: int = 128
    embedding_quantize: str | None = "4,32"
    max_seq_length: int = 384
    threads: int = 4

    def export_key(self) -> str:
        return f"gs{self.group_size}_emb{(self.embedding_quantize or 'none').replace(',', '-')}_seq{self.max_seq_length}"


def grid_size() -> int:
    return len(list(itertools.product(*SPACE.values())))


def load_golden(path: Path | str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Evaluators
# --------------------------------------------------------------------------
class RealEvaluator:
    """Export on the host (scripts/export_llama.sh), run golden prompts on the device."""

    def __init__(self, golden: list[dict], runner_cfg: RunnerConfig, out_dir: Path,
                 export_script: str = "scripts/export_llama.sh"):
        self.golden, self.rc, self.out, self.export_script = golden, runner_cfg, out_dir, export_script
        self.out.mkdir(parents=True, exist_ok=True)

    def _export(self, c: ETConfig) -> Path:
        pte = self.out / f"llama3_2_1b_{c.export_key()}.pte"
        if not pte.exists():
            subprocess.run(["bash", self.export_script, str(c.group_size), c.embedding_quantize or "none",
                            str(c.max_seq_length), str(pte)], check=True)
        return pte

    def __call__(self, c: ETConfig) -> Metrics:
        pte = self._export(c)
        rc = replace(self.rc, threads=c.threads, seq_len=c.max_seq_length, model=pte.name)
        if rc.mode == "adb":
            subprocess.run(["adb", "push", str(pte), f"{rc.workdir}/{pte.name}"], check=True, capture_output=True)
        else:
            rc = replace(rc, model=str(pte))
        runner, ok, ttft, tok_s = LlamaRunner(rc), 0, [], []
        for g in self.golden:
            text, st = runner.generate(chat_prompt(SYSTEM, build_event_text(g["event"], g["trend"])))
            d = extract_json(text)
            ok += int(bool(d) and d["action"] == g["expected_action"])
            if st.get("ttft_ms"):
                ttft.append(st["ttft_ms"])
            if st.get("decode_tok_s"):
                tok_s.append(st["decode_tok_s"])
        return Metrics(latency_ms=sum(ttft) / max(1, len(ttft)), memory_mb=pte.stat().st_size / 2**20,
                       accuracy=ok / len(self.golden), throughput=sum(tok_s) / max(1, len(tok_s)))


def mock_evaluator(c: ETConfig) -> Metrics:
    """Analytic stand-in shaped like Llama 3.2 1B on a 2+6-core Cortex-A78/A55 phone."""
    emb_mb = {None: 1002.0, "8,0": 501.0, "4,32": 282.0}[c.embedding_quantize]
    mem = 700.0 * (1 + 2.0 / c.group_size) + emb_mb
    thr = {2: 1.0, 4: 1.35, 6: 1.25}[c.threads]          # little cores hurt past 4
    gs_speed = {32: 0.86, 64: 0.95, 128: 1.0, 256: 1.02}[c.group_size]
    tok_s = 9.0 * thr * gs_speed * (1.0 + 0.15 * (1002.0 - emb_mb) / 1002.0)
    ttft = 2600.0 / (thr * gs_speed) * (c.max_seq_length / 384) ** 0.25
    acc = 0.94 - {32: 0.0, 64: 0.01, 128: 0.03, 256: 0.08}[c.group_size] - \
        {None: 0.0, "8,0": 0.0, "4,32": 0.02}[c.embedding_quantize]
    return Metrics(latency_ms=ttft, memory_mb=mem, accuracy=acc, throughput=tok_s)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
def search(evaluate: Callable[[ETConfig], Metrics], objective: Objective,
           start: ETConfig = ETConfig(), budget: int = 12) -> dict:
    cache: dict[ETConfig, Metrics] = {}

    def ev(c: ETConfig) -> Metrics:
        if c not in cache:
            cache[c] = evaluate(c)
        return cache[c]

    base = ev(start)
    score = lambda c: objective.score(ev(c), base)  # noqa: E731
    best, improved = start, True
    while improved and len(cache) < budget:
        improved = False
        for knob, values in SPACE.items():
            for v in values:
                cand = replace(best, **{knob: v})
                if cand in cache or len(cache) >= budget:
                    continue
                if score(cand) > score(best):
                    best, improved = cand, True
    feasible = objective.feasible(ev(best))
    return {
        "best": asdict(best), "best_metrics": ev(best).as_dict(), "feasible": feasible,
        "baseline": asdict(start), "baseline_metrics": base.as_dict(),
        "evaluations": len(cache), "grid_size": grid_size(),
        "history": [{"config": asdict(c), "metrics": m.as_dict(), "score": round(objective.score(m, base), 4),
                     "feasible": objective.feasible(m)} for c, m in cache.items()],
    }

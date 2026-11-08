"""Operator advisory: Llama 3.2 (ExecuTorch, on-device) with a rule fallback.

Event-driven: the LLM runs only for REJECT / ESCALATE parts, on a worker
thread, so it never limits the vision frame rate. Output is one JSON object;
anything unparsable falls back to the rule table and is labelled as such.

Runner: ExecuTorch's `llama_main` (examples/models/llama). Modes:
  local : binary on this device (Pi 5 / Termux on Android)
  adb   : binary on a phone, driven from a laptop over adb (development)
"""
from __future__ import annotations

import json
import queue
import re
import shlex
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ACTIONS = ("STOP_LINE", "SCRAP", "REWORK", "RECHECK", "CONTINUE")

SYSTEM = (
    "You are a quality-control assistant on a factory inspection line. Given one inspection "
    "event and recent line statistics, reply with ONE JSON object and nothing else: "
    '{"action": one of ' + "|".join(ACTIONS) + ', "root_cause": "<=20 words", '
    '"operator_check": "<=20 words, a concrete thing to inspect", "urgency": "low|medium|high"}. '
    "STOP_LINE only for critical defects that are repeating. SCRAP critical single parts. REWORK "
    "major repairable defects. RECHECK when the system was unsure (ESCALATE). CONTINUE for minor."
)

RULES = {
    ("critical", True): ("STOP_LINE", "Repeating critical defect suggests a process fault upstream.", "high"),
    ("critical", False): ("SCRAP", "Isolated critical defect.", "high"),
    ("major", True): ("REWORK", "Rising rate of major defects; check tooling and handling.", "medium"),
    ("major", False): ("REWORK", "Single major defect, likely handling damage.", "medium"),
    ("minor", True): ("RECHECK", "Minor defects are increasing; watch for drift.", "low"),
    ("minor", False): ("CONTINUE", "Minor cosmetic deviation within tolerance.", "low"),
}

CHECKS = {
    "scratch": "Inspect conveyor guides and gripper pads for burrs.",
    "missing_pad": "Check stencil apertures and paste volume at this location.",
    "contamination": "Check cleanliness of the handling station and air supply.",
    "crack": "Check press force and part temperature before forming.",
}


@dataclass
class RunnerConfig:
    mode: str = "off"                 # off | local | adb
    binary: str = "./llama_main"
    model: str = "llama3_2_1b.pte"
    tokenizer: str = "tokenizer.model"
    workdir: str = "/data/local/tmp/nirikshak"   # adb mode: device dir with all three files
    seq_len: int = 384
    threads: int = 4
    timeout_s: float = 60.0


def chat_prompt(system: str, user: str) -> str:
    # Llama 3.x instruct template; the runner adds BOS from the .pte metadata.
    return (f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n")


def parse_observer(text: str) -> dict:
    """ExecuTorch runner prints `PyTorchObserver {json}` with timing stats."""
    m = re.search(r"PyTorchObserver\s*(\{.*?\})", text)
    if not m:
        return {}
    s = json.loads(m.group(1))
    out = {"prompt_tokens": s.get("prompt_tokens"), "generated_tokens": s.get("generated_tokens")}
    try:
        out["ttft_ms"] = s["first_token_ms"] - s["inference_start_ms"]
        dec_ms = s["inference_end_ms"] - s["prompt_eval_end_ms"]
        out["decode_tok_s"] = 1000.0 * max(0, s["generated_tokens"] - 1) / dec_ms if dec_ms > 0 else None
        out["prefill_tok_s"] = 1000.0 * s["prompt_tokens"] / max(1e-6, s["prompt_eval_end_ms"] - s["inference_start_ms"])
    except (KeyError, TypeError):
        pass
    return out


def extract_json(text: str) -> dict | None:
    tail = text.split("assistant<|end_header_id|>")[-1]
    for m in re.finditer(r"\{[^{}]*\}", tail, re.S):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and str(d.get("action", "")).upper() in ACTIONS:
            d["action"] = d["action"].upper()
            return d
    return None


class LlamaRunner:
    def __init__(self, cfg: RunnerConfig):
        self.cfg = cfg

    @property
    def enabled(self) -> bool:
        return self.cfg.mode in ("local", "adb")

    def generate(self, prompt: str) -> tuple[str, dict]:
        c = self.cfg
        args = (f"--model_path={c.model} --tokenizer_path={c.tokenizer} --seq_len={c.seq_len} "
                f"--temperature=0 --cpu_threads={c.threads}")
        if c.mode == "local":
            cmd = [c.binary, *shlex.split(args), f"--prompt={prompt}"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=c.timeout_s)
        else:  # adb: ship the prompt as a file so quoting can't break it
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
                f.write(prompt)
            subprocess.run(["adb", "push", f.name, f"{c.workdir}/prompt.txt"], check=True,
                           capture_output=True, timeout=c.timeout_s)
            Path(f.name).unlink(missing_ok=True)
            sh = f'cd {c.workdir} && {c.binary} {args} --prompt="$(cat prompt.txt)"'
            res = subprocess.run(["adb", "shell", sh], capture_output=True, text=True, timeout=c.timeout_s)
        out = res.stdout + "\n" + res.stderr
        return out, parse_observer(out)


def build_event_text(event: dict, trend: dict) -> str:
    return json.dumps({"event": event, "line_trend": trend}, separators=(",", ":"))


def rule_advice(event: dict, trend: dict) -> dict:
    sev = event.get("severity", "minor")
    rising = bool(trend.get("prev_defect_rate") is not None
                  and trend["defect_rate"] > max(0.05, 1.5 * trend["prev_defect_rate"]))
    if event.get("decision") == "ESCALATE":
        action, cause, urg = "RECHECK", "Model not confident; borderline anomaly score.", "medium"
    else:
        action, cause, urg = RULES[(sev, rising)]
    return {"action": action, "root_cause": cause, "urgency": urg,
            "operator_check": CHECKS.get(event.get("label", ""), "Inspect the highlighted region under magnification.")}


class Advisor:
    def __init__(self, runner: LlamaRunner | None = None):
        self.runner = runner
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._t: threading.Thread | None = None

    def advise(self, event: dict, trend: dict) -> dict:
        t0 = time.perf_counter()
        if self.runner and self.runner.enabled:
            try:
                text, stats = self.runner.generate(chat_prompt(SYSTEM, build_event_text(event, trend)))
                d = extract_json(text)
                if d:
                    return {**d, "source": "llama-executorch", "stats": stats,
                            "latency_ms": round((time.perf_counter() - t0) * 1e3, 1)}
                fallback_reason = "unparsable LLM output"
            except (subprocess.SubprocessError, OSError) as e:
                fallback_reason = f"runner error: {type(e).__name__}"
        else:
            fallback_reason = "LLM disabled"
        return {**rule_advice(event, trend), "source": "rules", "fallback_reason": fallback_reason,
                "latency_ms": round((time.perf_counter() - t0) * 1e3, 1)}

    # --- async worker -----------------------------------------------------
    def start(self, on_result: Callable[[int, dict], None]) -> None:
        def loop() -> None:
            while True:
                item = self._q.get()
                if item is None:
                    return
                row_id, event, trend = item
                try:
                    on_result(row_id, self.advise(event, trend))
                except Exception as e:  # keep the worker alive; log and move on
                    on_result(row_id, {**rule_advice(event, trend), "source": "rules",
                                       "fallback_reason": f"advisor error: {e!r}"})
                finally:
                    self._q.task_done()
        self._t = threading.Thread(target=loop, daemon=True, name="advisor")
        self._t.start()

    def submit(self, row_id: int, event: dict, trend: dict) -> bool:
        try:
            self._q.put_nowait((row_id, event, trend))
            return True
        except queue.Full:  # never block the line; operator still sees the decision
            return False

    def drain(self) -> None:
        if self._t:
            self._q.join()

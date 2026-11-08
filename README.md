# Nirikshak: on-device defect inspection with a Llama QC assistant

**Bharat AI-SoC Challenge 2026–27 · Problem Statement 5: Real-Time Industrial Defect Detection & Quality-Control Assistant**

Nirikshak inspects parts on an Arm device, flags defects without needing a labelled defect set, says when it is unsure, and gives the operator an action from Llama 3.2 running on-device through ExecuTorch. Nothing goes to the cloud.

```
camera ─► localize (training-free)        lattice FFT → compare each repeat with its 24 neighbours
       │                                  (PCB, wafer, fabric, grids) · texture residual otherwise
       ├► classify defect type            MobileNetV3 ONNX via OpenCV DNN (optional)
       ├► grade severity                  critical / major / minor
       ├► calibrated gate                 PASS · REJECT · ESCALATE (system not sure → human)
       ├► SQLite log + dashboard          http://<device>:8000
       └► advisor (REJECT/ESCALATE only, async)
              Llama 3.2 1B Instruct · ExecuTorch XNNPACK · quant config picked by Anvil for this device
              → {action, root_cause, operator_check, urgency}; rule fallback if output is invalid
```

## PS5 requirements → where they are met

| Requirement | Implementation |
|---|---|
| Detect surface defects, cracks, anomalies | `nirikshak/localize.py`: lattice-neighbour residual (periodic) / median background residual (stochastic), robust z-scores |
| Classify defect type and severity | `classify.py` (ONNX, trained by `scripts/train_classifier.py`), `gate.grade` |
| Alert operators with actionable recommendations | `advisor.py` + dashboard (`server.py`) |
| Log inspection results | `store.py` (SQLite: decision, score, regions, latency, advice) |
| Real time at the edge, no cloud | everything runs on the device; LLM is event-driven so it never limits frame rate |
| **Llama + ExecuTorch on-device (required)** | `scripts/export_llama.sh` (8da4w XNNPACK `.pte`), `scripts/build_runner_android.sh` (`llama_main`), `advisor.LlamaRunner` |
| Separate vision/LLM pipelines documented | Vision: OpenCV on CPU. LLM: ExecuTorch on CPU, separate worker thread. No NPU is used on the phone. |

## What is different

1. **No defect training data needed to start.** DriftSense's lattice stage recovers the repeat vectors of a periodic part; a defect exists in one repeat only, so the median of the neighbours rejects it. New product, no retraining.
2. **It knows when it is unsure.** Thresholds are fitted to target escape and false-reject rates; the Wilson 95% upper bounds are reported. Parts in between go to a human (ESCALATE), not a guess.
3. **The LLM is tuned to the device.** Anvil searches ExecuTorch export knobs (group size, embedding quant, context, threads) under a task-quality floor measured on `data/golden_advice.jsonl`, in a fraction of the grid's evaluations.

## Quick start (no dataset, no device)

```bash
pip install -e .            # numpy + opencv only
nirikshak demo --n 150      # synthetic PCB-like parts: calibrate, run, score escapes/false rejects
nirikshak demo --serve 8000 # same, with the dashboard
pytest -q                   # 30 tests (pipeline, gate, advisor with a stand-in runner, dashboard, Anvil)
```

## On the device

**Android phone (Termux), e.g. iQOO Z7s / Snapdragon 695** or **Raspberry Pi 5**:

```bash
bash scripts/setup_device.sh
nirikshak --config configs/default.toml run --source http://127.0.0.1:8080/shot.jpg --serve 8000   # phone: IP Webcam app
nirikshak --config configs/default.toml run --source 0 --serve 8000                                   # Pi: USB camera
```

**Llama 3.2 1B via ExecuTorch** (export on the laptop, run on the device):

```bash
export ET_ROOT=~/executorch LLAMA_DIR=~/Llama-3.2-1B-Instruct/original ANDROID_NDK=~/android-ndk
bash scripts/export_llama.sh 64 4,32 384 models/llm/llama3_2_1b.pte
bash scripts/build_runner_android.sh            # cross-compiles and pushes llama_main
# set [runner] mode = "local" (on device) or "adb" (laptop drives the phone) in configs/default.toml
nirikshak --config configs/default.toml advise --event '{"decision":"REJECT","label":"scratch","severity":"critical"}'
```

**Tune the LLM for the device (Anvil):**

```bash
nirikshak --config configs/default.toml tune-llm --budget 12 --quality-floor 0.9 --size-budget-mb 1200
nirikshak tune-llm --mock      # same search against an analytic model, no hardware
```

**Calibrate on real data** (e.g. MVTec AD `grid`, `carpet`, `tile`; DeepPCB):

```bash
nirikshak calibrate --good data/mvtec/grid/test/good --defect data/mvtec/grid/test --out calibration.json
```

**Train the defect-type classifier** (laptop GPU; NEU-CLS + MVTec/DeepPCB crops in ImageFolder layout):

```bash
pip install -e .[train]
python scripts/train_classifier.py --data data/cls --out models/defect_cls.onnx
```

## Current results (synthetic sanity set, laptop CPU)

| Metric | Value |
|---|---|
| Parts | 150 (43 defective) |
| False rejects | 0 |
| Escapes | 3 (small, low-contrast contamination) |
| Localize latency p50 | ~300 ms at 512 px |
| Anvil (mock) | feasible config in 9 of 72 evaluations |

Device numbers (Z7s / Pi 5) and MVTec/NEU results are the next step and will be added from measured runs only.

## Layout

```
nirikshak/   localize · lattice (from DriftSense) · classify · gate · advisor · llm_tune · store · server · cli · synth
anvil/       quantization-search agent (from adaptive-llm-quantization)
scripts/     export_llama.sh · build_runner_android.sh · setup_device.sh · train_classifier.py
data/        golden_advice.jsonl (advisor quality set)
configs/     default.toml
```

Builds on my earlier work: [DriftSense](https://github.com/HUNT-001/DriftSense) (lattice localisation, calibrated confidence) and [adaptive-llm-quantization](https://github.com/HUNT-001/adaptive-llm-quantization) (Anvil).

ExecuTorch commands follow `examples/models/llama/README.md`. Pin the ExecuTorch tag you build with; newer tags move the export entry point to `extension/llm/export`.

MIT licence.

#!/usr/bin/env bash
# On-device setup. Pi 5 (Raspberry Pi OS 64-bit) or Android Termux.
set -euo pipefail
if command -v termux-info >/dev/null 2>&1; then
  pkg install -y python opencv-python python-numpy git
else
  sudo apt-get update && sudo apt-get install -y python3-opencv python3-numpy python3-venv
  python3 -m venv --system-site-packages .venv && . .venv/bin/activate   # PEP 668 on Pi OS
fi
python3 -m pip install -e . --no-deps
echo "ok: run 'nirikshak demo' to self-test"

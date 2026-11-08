"""Defect-type classifier on region crops.

Runs an ONNX model through OpenCV DNN (works on Pi OS, Termux and laptops with
no extra runtime). Train/export with scripts/train_classifier.py. With no model
the label comes from shape only ("linear" / "blob"), marked as such.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .localize import Region


class DefectClassifier:
    def __init__(self, onnx_path: str | None, labels_path: str | None = None, size: int = 224):
        self.net = None
        self.labels: list[str] = []
        self.size = size
        if onnx_path and Path(onnx_path).exists():
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
            lp = Path(labels_path) if labels_path else Path(onnx_path).with_suffix(".labels.json")
            self.labels = json.loads(lp.read_text())

    @property
    def available(self) -> bool:
        return self.net is not None

    def crop(self, img: np.ndarray, r: Region, ctx: float = 2.0, min_px: int = 48) -> np.ndarray:
        x, y, w, h = r.bbox
        cx, cy = x + w / 2, y + h / 2
        half = max(min_px, ctx * max(w, h)) / 2
        H, W = img.shape[:2]
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
        return img[y0:y1, x0:x1]

    def __call__(self, img: np.ndarray, regions: list[Region], top_k: int = 3) -> None:
        for r in regions[:top_k]:
            if self.net is None:
                r.label, r.label_prob = ("linear" if r.elongation >= 4 else "blob"), 0.0
                continue
            c = self.crop(img, r)
            if c.ndim == 2:
                c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
            blob = cv2.dnn.blobFromImage(c, 1 / 255.0, (self.size, self.size), swapRB=True)
            blob = (blob - np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)) / \
                np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            self.net.setInput(blob.astype(np.float32))
            logits = self.net.forward().ravel()
            p = np.exp(logits - logits.max())
            p /= p.sum()
            i = int(p.argmax())
            r.label, r.label_prob = self.labels[i], float(p[i])

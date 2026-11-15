from __future__ import annotations

import io
import math
from collections.abc import Sequence

import numpy as np
from PIL import Image

from packages.vision.contracts import BoundingBox, CameraFrame, EmbeddingVector


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two equal-length vectors in [-1.0, 1.0]."""
    if len(a) != len(b) or not a:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))


class LocalHistogramVisualEncoder:
    """
    Lightweight, deterministic local visual encoder (64-dim L2-normalized embedding).

    Combines coarse RGB color histograms and spatial 2x2 grid statistics so
    teachable object recognition works offline out of the box, while keeping the
    VisualEncoder contract ready for an ONNX MobileNet/CLIP encoder upgrade.
    """

    encoder_version = "local-hist-v1"

    def encode(self, frame: CameraFrame, box: BoundingBox | None = None) -> EmbeddingVector:
        img = Image.open(io.BytesIO(frame.jpeg_bytes)).convert("RGB")
        if box is not None:
            w, h = img.size
            left = max(0, min(w - 1, int(box.x1 * w)))
            top = max(0, min(h - 1, int(box.y1 * h)))
            right = max(left + 1, min(w, int(box.x2 * w)))
            bottom = max(top + 1, min(h, int(box.y2 * h)))
            img = img.crop((left, top, right, bottom))

        img = img.resize((64, 64), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0

        # 16-bin histogram per R, G, B channel = 48 dims
        feats: list[float] = []
        for ch in range(3):
            hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0.0, 1.0))
            feats.extend(hist.astype(np.float32).tolist())

        # 2x2 spatial mean RGB + luminance = 16 dims (total 64 dims)
        for gy in (0, 32):
            for gx in (0, 32):
                patch = arr[gy : gy + 32, gx : gx + 32, :]
                mean_rgb = patch.mean(axis=(0, 1))
                luma = float(0.299 * mean_rgb[0] + 0.587 * mean_rgb[1] + 0.114 * mean_rgb[2])
                feats.extend([float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2]), luma])

        vec = np.asarray(feats, dtype=np.float32)
        norm = float(math.sqrt(float(np.sum(vec * vec))))
        if norm > 1e-8:
            vec = vec / norm

        return EmbeddingVector(
            values=[round(float(x), 6) for x in vec.tolist()], encoder_version=self.encoder_version
        )

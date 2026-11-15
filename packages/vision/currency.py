from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np
from PIL import Image

from packages.vision.contracts import (
    BoundingBox,
    CameraFrame,
    CurrencyAnalysisResult,
    CurrencyNoteCandidate,
)
from packages.vision.detector import assess_frame_lighting

SUPPORTED_INR_DENOMINATIONS = (10, 20, 50, 100, 200, 500, 2000)

# Approximate dominant RGB centroids (0..1) of modern Mahatma Gandhi (New) series INR notes
_INR_COLOR_PROFILES: dict[int, tuple[float, float, float]] = {
    10: (0.56, 0.42, 0.31),  # Chocolate brown
    20: (0.64, 0.68, 0.34),  # Greenish-yellow
    50: (0.35, 0.67, 0.72),  # Fluorescent blue
    100: (0.55, 0.46, 0.65),  # Lavender
    200: (0.83, 0.56, 0.24),  # Bright orange-yellow
    500: (0.53, 0.55, 0.52),  # Stone grey
    2000: (0.68, 0.36, 0.56),  # Magenta
}


def _box_iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 1e-8 else 0.0


class INRCurrencyAnalyzer:
    """
    Indian banknote denomination analyzer with multi-frame de-duplication,
    quality/lighting checks, confidence abstention, and non-authentication safety.
    """

    model_version = "inr-color-geom-v1"

    def __init__(self, confidence_threshold: float = 0.72) -> None:
        self.confidence_threshold = confidence_threshold

    def analyze(
        self,
        frames: Sequence[CameraFrame],
        injected_candidates: Sequence[CurrencyNoteCandidate] | None = None,
    ) -> CurrencyAnalysisResult:
        if not frames and not injected_candidates:
            return CurrencyAnalysisResult(
                notes=[],
                distinct_count=0,
                total_inr=0,
                abstained=True,
                uncertainty_reason="no_frames",
                guidance="No camera frame is available. Please start the camera and hold the note steady.",
            )

        if frames:
            _, lighting_issue = assess_frame_lighting(frames[-1])
            if lighting_issue is not None:
                return CurrencyAnalysisResult(
                    notes=[],
                    distinct_count=0,
                    total_inr=0,
                    abstained=True,
                    uncertainty_reason="poor_lighting",
                    guidance=lighting_issue,
                )

        raw_candidates: list[CurrencyNoteCandidate] = list(injected_candidates or [])
        if not raw_candidates and frames:
            for frame in frames:
                raw_candidates.extend(self._analyze_single_frame(frame))

        if not raw_candidates:
            return CurrencyAnalysisResult(
                notes=[],
                distinct_count=0,
                total_inr=0,
                abstained=True,
                uncertainty_reason="no_note_detected",
                guidance=(
                    "I could not clearly see an Indian banknote. "
                    "Hold the note flat in good light, about 20 to 30 centimeters from the camera."
                ),
            )

        # De-duplicate across frames using track_id or spatial overlap (IoU > 0.45)
        deduped = self._deduplicate_candidates(raw_candidates)
        confident = [c for c in deduped if c.confidence >= self.confidence_threshold]

        if not confident:
            return CurrencyAnalysisResult(
                notes=[],
                distinct_count=0,
                total_inr=0,
                abstained=True,
                uncertainty_reason="low_confidence",
                guidance=(
                    "I am not confident enough to read the denomination clearly. "
                    "Please flatten the note, reduce glare, or show the other side."
                ),
            )

        total = sum(n.denomination_inr for n in confident)
        count = len(confident)
        summary = ", ".join(f"Rs {n.denomination_inr}" for n in confident)
        guidance = (
            f"Detected {count} {'note' if count == 1 else 'notes'}: {summary}. "
            f"Estimated total: {total} rupees."
        )
        if len(confident) < len(deduped):
            guidance += " Some overlapping or blurry notes could not be read reliably."

        return CurrencyAnalysisResult(
            notes=confident,
            distinct_count=count,
            total_inr=total,
            abstained=False,
            uncertainty_reason=None,
            guidance=guidance,
        )

    def _analyze_single_frame(self, frame: CameraFrame) -> list[CurrencyNoteCandidate]:
        img = Image.open(io.BytesIO(frame.jpeg_bytes)).convert("RGB")
        small = img.resize((64, 48), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0

        # Sample center region where a held banknote typically appears
        center = arr[10:38, 12:52, :]
        std_rgb = float(center.std())
        if std_rgb < 0.03:
            # Uniform blank/flat background, unlikely to be a textured banknote
            return []

        mean_rgb = center.mean(axis=(0, 1))
        best_denom = 0
        best_dist = 999.0
        for denom, proto in _INR_COLOR_PROFILES.items():
            dist = float(np.linalg.norm(mean_rgb - np.asarray(proto, dtype=np.float32)))
            if dist < best_dist:
                best_dist = dist
                best_denom = denom

        # Convert color distance to a conservative confidence score
        confidence = max(0.0, min(0.95, 1.0 - (best_dist / 0.45)))
        if best_denom not in SUPPORTED_INR_DENOMINATIONS or confidence < 0.45:
            return []

        return [
            CurrencyNoteCandidate(
                denomination_inr=best_denom,
                confidence=round(confidence, 3),
                box=BoundingBox(x1=0.18, y1=0.20, x2=0.82, y2=0.78),
                track_id="center-note",
            )
        ]

    @staticmethod
    def _deduplicate_candidates(
        candidates: Sequence[CurrencyNoteCandidate],
    ) -> list[CurrencyNoteCandidate]:
        kept: list[CurrencyNoteCandidate] = []
        seen_tracks: dict[str, int] = {}

        for cand in candidates:
            if cand.track_id and cand.track_id in seen_tracks:
                idx = seen_tracks[cand.track_id]
                if cand.confidence > kept[idx].confidence:
                    kept[idx] = cand
                continue

            matched_idx: int | None = None
            for i, existing in enumerate(kept):
                if _box_iou(existing.box, cand.box) > 0.45:
                    matched_idx = i
                    break

            if matched_idx is not None:
                if cand.confidence > kept[matched_idx].confidence:
                    kept[matched_idx] = cand
            else:
                if cand.track_id:
                    seen_tracks[cand.track_id] = len(kept)
                kept.append(cand)

        return kept

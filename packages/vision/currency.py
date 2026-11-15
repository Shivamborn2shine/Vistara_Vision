from __future__ import annotations

import colorsys
import io
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from packages.vision.contracts import (
    BoundingBox,
    CameraFrame,
    CurrencyAnalysisResult,
    CurrencyNoteCandidate,
)
from packages.vision.detector import assess_frame_lighting

SUPPORTED_INR_DENOMINATIONS = (10, 20, 50, 100, 200, 500, 2000)
_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "models" / "weights"
_YOLO_CURRENCY_PATH = _WEIGHTS_DIR / "yolov8n_currency_best.onnx"
_CNN_CURRENCY_PATH = _WEIGHTS_DIR / "mobilenetv3_currency.onnx"
_HF_CLASSES = (10, 100, 20, 200, 2000, 50, 500)

# Normalized RGB ratios (r/sum, g/sum, b/sum) and HSV hue ranges (0..360) for modern INR notes
_INR_PROFILES: dict[int, dict[str, Any]] = {
    10: {"rgb_ratio": (0.40, 0.32, 0.28), "hue": (10.0, 38.0), "min_sat": 0.14},
    20: {"rgb_ratio": (0.37, 0.40, 0.23), "hue": (52.0, 90.0), "min_sat": 0.16},
    50: {"rgb_ratio": (0.22, 0.38, 0.40), "hue": (170.0, 212.0), "min_sat": 0.18},
    100: {"rgb_ratio": (0.34, 0.31, 0.35), "hue": (210.0, 325.0), "min_sat": 0.05},
    200: {"rgb_ratio": (0.48, 0.34, 0.18), "hue": (20.0, 48.0), "min_sat": 0.28},
    500: {"rgb_ratio": (0.34, 0.34, 0.32), "hue": (0.0, 360.0), "min_sat": 0.0},
    2000: {"rgb_ratio": (0.41, 0.25, 0.34), "hue": (290.0, 350.0), "min_sat": 0.18},
}

_DENOM_RE = re.compile(r"(?:₹|rs\.?|inr)?\s*\b(10|20|50|100|200|500|2000)\b", re.IGNORECASE)
_CURRENCY_CONTEXT_RE = re.compile(
    r"(₹|reserve\s*bank|rbi|india|rupee|भारत|रुपये)", re.IGNORECASE
)


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


def _rgb_to_hsv_deg(r: float, g: float, b: float) -> tuple[float, float, float]:
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360.0, s, v


class INRCurrencyAnalyzer:
    """
    Indian banknote denomination analyzer combining:
    1. Multi-orientation Local OCR denomination + RBI/rupee text matching (RapidOCR)
    2. Detector-guided & multi-window HSV + normalized-RGB banknote classification
    3. Multi-frame de-duplication, lighting/confidence abstention, and non-authentication safety
    """

    model_version = "inr-yolo-cnn-ocr-hsv-v4"

    def __init__(self, confidence_threshold: float = 0.68) -> None:
        self.confidence_threshold = confidence_threshold
        self._ocr: Any = None
        self._ocr_checked = False
        self._yolo: Any = None
        self._cnn: Any = None
        self._models_checked = False

    def _init_onnx_models(self) -> None:
        if self._models_checked:
            return
        self._models_checked = True
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]

            if _YOLO_CURRENCY_PATH.exists():
                self._yolo = ort.InferenceSession(
                    str(_YOLO_CURRENCY_PATH), providers=["CPUExecutionProvider"]
                )
            if _CNN_CURRENCY_PATH.exists():
                self._cnn = ort.InferenceSession(
                    str(_CNN_CURRENCY_PATH), providers=["CPUExecutionProvider"]
                )
        except Exception:
            self._yolo = None
            self._cnn = None

    def _detect_yolo_note_boxes(self, img: Image.Image) -> list[tuple[BoundingBox, int, float]]:
        self._init_onnx_models()
        if self._yolo is None:
            return []
        try:
            scale = min(416.0 / img.width, 416.0 / img.height)
            nw, nh = int(round(img.width * scale)), int(round(img.height * scale))
            pad_x, pad_y = (416 - nw) // 2, (416 - nh) // 2
            canvas = Image.new("RGB", (416, 416), (114, 114, 114))
            canvas.paste(img.resize((nw, nh), Image.Resampling.BILINEAR), (pad_x, pad_y))
            arr = np.asarray(canvas, dtype=np.float32) / 255.0
            inp = np.transpose(arr, (2, 0, 1))[None, ...]
            out = self._yolo.run(None, {self._yolo.get_inputs()[0].name: inp})[0][0].T
            scores = out[:, 4:]
            best_cls = scores.argmax(axis=1)
            best_conf = scores.max(axis=1)
            hits: list[tuple[BoundingBox, int, float]] = []
            for idx in np.where(best_conf >= 0.45)[0]:
                cx, cy, bw, bh = [float(v) for v in out[idx, :4]]
                x1 = max(0.0, min(1.0, ((cx - bw / 2.0) - pad_x) / nw))
                y1 = max(0.0, min(1.0, ((cy - bh / 2.0) - pad_y) / nh))
                x2 = max(x1 + 0.02, min(1.0, ((cx + bw / 2.0) - pad_x) / nw))
                y2 = max(y1 + 0.02, min(1.0, ((cy + bh / 2.0) - pad_y) / nh))
                hits.append(
                    (
                        BoundingBox(round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)),
                        _HF_CLASSES[int(best_cls[idx])],
                        float(best_conf[idx]),
                    )
                )
            # NMS by IoU
            hits.sort(key=lambda h: h[2], reverse=True)
            kept: list[tuple[BoundingBox, int, float]] = []
            for b, d, c in hits:
                if all(_box_iou(b, kb) < 0.45 for kb, _, _ in kept):
                    kept.append((b, d, c))
            return kept
        except Exception:
            return []

    def _classify_with_cnn(self, crop: Image.Image) -> tuple[int, float] | None:
        self._init_onnx_models()
        if self._cnn is None:
            return None
        try:
            resized = crop.resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            inp = np.transpose(arr, (2, 0, 1))[None, ...]
            logits = self._cnn.run(None, {self._cnn.get_inputs()[0].name: inp})[0][0]
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            idx = int(probs.argmax())
            return _HF_CLASSES[idx], float(probs[idx])
        except Exception:
            return None

    def _get_ocr(self) -> Any:
        if self._ocr_checked:
            return self._ocr
        self._ocr_checked = True
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]

            self._ocr = RapidOCR()
        except Exception:
            self._ocr = None
        return self._ocr

    def analyze(
        self,
        frames: Sequence[CameraFrame],
        injected_candidates: Sequence[CurrencyNoteCandidate] | None = None,
        candidate_boxes: Sequence[BoundingBox] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> CurrencyAnalysisResult:
        if cancel_check and cancel_check():
            return CurrencyAnalysisResult(
                notes=[],
                distinct_count=0,
                total_inr=0,
                abstained=True,
                uncertainty_reason="cancelled",
                guidance="Currency check superseded by a newer request.",
            )
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
                if cancel_check and cancel_check():
                    break
                raw_candidates.extend(
                    self._analyze_single_frame(
                        frame,
                        candidate_boxes=candidate_boxes,
                        cancel_check=cancel_check,
                    )
                )

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
                    "Please hold the note slightly closer and flatter in good light."
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

    def _analyze_single_frame(
        self,
        frame: CameraFrame,
        candidate_boxes: Sequence[BoundingBox] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[CurrencyNoteCandidate]:
        if cancel_check and cancel_check():
            return []
        img = Image.open(io.BytesIO(frame.jpeg_bytes)).convert("RGB")
        # Prefer YOLOv8n-currency bounding boxes when available
        yolo_hits = self._detect_yolo_note_boxes(img)
        if cancel_check and cancel_check():
            return []
        merged_boxes: list[BoundingBox] = [b for b, _, _ in yolo_hits]
        for b in candidate_boxes or []:
            if all(_box_iou(b, mb) < 0.45 for mb in merged_boxes):
                merged_boxes.append(b)

        ocr_hits = self._detect_via_ocr(img, merged_boxes or None, cancel_check=cancel_check)
        if ocr_hits or (cancel_check and cancel_check()):
            return ocr_hits

        if merged_boxes:
            box_results: list[CurrencyNoteCandidate] = []
            for idx, box in enumerate(merged_boxes):
                cand = self._classify_crop_hsv(img, box, track_id=f"det-{idx}", allow_grey_500=True)
                if cand is not None:
                    box_results.append(cand)
            return box_results

        windows = [
            BoundingBox(0.20, 0.22, 0.80, 0.78),
            BoundingBox(0.32, 0.55, 0.68, 0.90),
        ]
        best: CurrencyNoteCandidate | None = None
        for box in windows:
            cand = self._classify_crop_hsv(img, box, track_id="center-note", allow_grey_500=False)
            if cand is not None and (best is None or cand.confidence > best.confidence):
                best = cand
        return [best] if best is not None else []

    def _detect_via_ocr(
        self,
        img: Image.Image,
        candidate_boxes: Sequence[BoundingBox] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[CurrencyNoteCandidate]:
        ocr = self._get_ocr()
        if ocr is None:
            return []

        crops: list[tuple[Image.Image, BoundingBox]] = []
        w, h = img.size
        for box in candidate_boxes or []:
            crop = img.crop((int(box.x1 * w), int(box.y1 * h), int(box.x2 * w), int(box.y2 * h)))
            if crop.width > 24 and crop.height > 24:
                crops.append((crop, box))
        if not crops:
            crops.append((img, BoundingBox(0.15, 0.15, 0.85, 0.85)))

        found: list[CurrencyNoteCandidate] = []
        for crop_img, box in crops:
            if cancel_check and cancel_check():
                return []
            enhanced = ImageEnhance.Contrast(crop_img).enhance(1.4)
            for angle in (0, 90):
                if cancel_check and cancel_check():
                    return []
                rot = enhanced.rotate(angle, expand=True) if angle else enhanced
                if rot.width < 480:
                    rot = rot.resize((rot.width * 2, rot.height * 2), Image.Resampling.BILINEAR)
                try:
                    res, _ = ocr(np.asarray(rot))
                except Exception:
                    continue
                if not res:
                    continue
                joined = " ".join(str(item[1]) for item in res)
                has_context = bool(_CURRENCY_CONTEXT_RE.search(joined))
                for item in res:
                    txt = str(item[1]).strip()
                    conf = float(item[2])
                    for m in _DENOM_RE.finditer(txt):
                        denom = int(m.group(1))
                        if denom in SUPPORTED_INR_DENOMINATIONS:
                            score = min(0.96, conf if has_context else conf * 0.90)
                            found.append(
                                CurrencyNoteCandidate(
                                    denomination_inr=denom,
                                    confidence=round(score, 3),
                                    box=box,
                                    track_id=f"ocr-{denom}",
                                )
                            )
                if found:
                    return found
        return found

    def _classify_crop_hsv(
        self,
        img: Image.Image,
        box: BoundingBox,
        track_id: str,
        allow_grey_500: bool,
    ) -> CurrencyNoteCandidate | None:
        w, h = img.size
        # Trim 15% from each edge to reduce finger/background contamination
        bw = box.x2 - box.x1
        bh = box.y2 - box.y1
        ix1 = box.x1 + 0.15 * bw
        iy1 = box.y1 + 0.15 * bh
        ix2 = box.x2 - 0.15 * bw
        iy2 = box.y2 - 0.15 * bh
        crop = img.crop(
            (
                max(0, int(ix1 * w)),
                max(0, int(iy1 * h)),
                min(w, int(ix2 * w)),
                min(h, int(iy2 * h)),
            )
        )
        if crop.width < 12 or crop.height < 12:
            return None

        arr = np.asarray(crop.resize((64, 48), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        if float(arr.std()) < 0.045:
            return None

        med_rgb = np.median(arr, axis=(0, 1))
        r, g, b = float(med_rgb[0]), float(med_rgb[1]), float(med_rgb[2])
        rgb_sum = r + g + b
        if rgb_sum < 0.15:
            return None

        rgb_ratio = np.asarray([r / rgb_sum, g / rgb_sum, b / rgb_sum], dtype=np.float32)
        rr, rg, rb = float(rgb_ratio[0]), float(rgb_ratio[1]), float(rgb_ratio[2])
        hue, sat, _ = _rgb_to_hsv_deg(r, g, b)

        best_denom = 0
        best_score = 0.0

        for denom, prof in _INR_PROFILES.items():
            if denom == 500 and not allow_grey_500:
                continue
            proto_ratio = np.asarray(prof["rgb_ratio"], dtype=np.float32)
            ratio_dist = float(np.linalg.norm(rgb_ratio - proto_ratio))
            score = max(0.0, 1.0 - (ratio_dist / 0.14))

            h_lo, h_hi = prof["hue"]
            if denom == 500:
                if sat > 0.10 or rb > rg + 0.015:
                    score *= 0.45
            elif denom == 100:
                # Lavender: Blue is noticeably higher than Green (violet/lavender)
                if (rb > rg + 0.015 and rb > rr - 0.02) or (
                    h_lo <= hue <= h_hi and sat >= prof["min_sat"] and rb > rg
                ):
                    score = min(0.95, score + 0.22)
                else:
                    score *= 0.55
            elif denom == 50:
                # Cyan-blue: Green is strong alongside Blue, Red is low
                if h_lo <= hue <= 208.0 and sat >= prof["min_sat"] and rg >= rb * 0.96:
                    score = min(0.95, score + 0.18)
                else:
                    score *= 0.55
            else:
                if h_lo <= hue <= h_hi and sat >= prof["min_sat"]:
                    score = min(0.95, score + 0.18)
                elif sat < prof["min_sat"]:
                    score *= 0.60

            if score > best_score:
                best_score = score
                best_denom = denom

        if best_denom not in SUPPORTED_INR_DENOMINATIONS or best_score < 0.55:
            return None

        cnn_res = self._classify_with_cnn(crop)
        if cnn_res is not None:
            cnn_denom, cnn_conf = cnn_res
            if cnn_denom == best_denom and cnn_conf >= 0.65:
                best_score = min(0.97, max(best_score, cnn_conf) + 0.08)

        return CurrencyNoteCandidate(
            denomination_inr=best_denom,
            confidence=round(best_score, 3),
            box=box,
            track_id=track_id,
        )

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

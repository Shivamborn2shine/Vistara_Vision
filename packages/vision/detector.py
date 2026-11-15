from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from packages.vision.coco_labels import MSCOCO_90_LABELS
from packages.vision.contracts import BoundingBox, CameraFrame, Detection


def describe_relative_position(box: BoundingBox) -> str:
    """Describe qualitative position in the frame without fabricating room coordinates."""
    cx = box.center_x
    cy = box.center_y

    horiz = "on the left" if cx < 0.36 else ("on the right" if cx > 0.64 else "in the center")
    vert = "upper area" if cy < 0.35 else ("lower area" if cy > 0.68 else "middle area")
    return f"{horiz}, {vert}"


def assess_frame_lighting(frame: CameraFrame) -> tuple[float, str | None]:
    """Return mean luminance [0..1] and an optional guidance message for dark/overexposed frames."""
    img = Image.open(io.BytesIO(frame.jpeg_bytes)).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean_luma = float(arr.mean())
    if mean_luma < 0.14:
        return (
            mean_luma,
            "The scene looks too dark. Please turn on a light or move closer to a light source.",
        )
    if mean_luma > 0.94:
        return (
            mean_luma,
            "The image has strong glare or overexposure. Try tilting the camera slightly.",
        )
    return mean_luma, None


class HeuristicObjectDetector:
    """
    Baseline local detector combining YCbCr person/face detection and salient
    foreground region analysis.

    Runs in ~5ms on CPU with zero external weight downloads, and can be swapped
    for an ONNX YOLO/MediaPipe detector via the ObjectDetector protocol.
    """

    model_version = "local-person-salient-v2"

    def detect(self, frame: CameraFrame) -> list[Detection]:
        img = Image.open(io.BytesIO(frame.jpeg_bytes)).convert("RGB")
        small = img.resize((64, 48), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0

        detections: list[Detection] = []
        person_det = self._detect_person(small)
        if person_det is not None:
            detections.append(person_det)

        # Identify salient foreground regions relative to frame median color
        median_rgb = np.median(arr, axis=(0, 1))
        diff = np.linalg.norm(arr - median_rgb, axis=2)
        mask = diff > 0.22

        # Exclude the detected person region so we also spot held/nearby objects
        if person_det is not None:
            px1 = int(person_det.box.x1 * 64)
            py1 = int(person_det.box.y1 * 48)
            px2 = int(person_det.box.x2 * 64)
            py2 = int(person_det.box.y2 * 48)
            mask[max(0, py1) : min(48, py2), max(0, px1) : min(64, px2)] = False

        if np.any(mask):
            ys, xs = np.where(mask)
            x1 = float(xs.min()) / 64.0
            y1 = float(ys.min()) / 48.0
            x2 = float(xs.max() + 1) / 64.0
            y2 = float(ys.max() + 1) / 48.0

            if (x2 - x1) > 0.08 and (y2 - y1) > 0.08:
                region = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
                mean_rgb = region.mean(axis=(0, 1))
                label = self._classify_salient_region(mean_rgb, x2 - x1, y2 - y1)
                detections.append(
                    Detection(
                        label=label,
                        box=BoundingBox(
                            x1=round(x1, 3), y1=round(y1, 3), x2=round(x2, 3), y2=round(y2, 3)
                        ),
                        confidence=0.72,
                        model_version=self.model_version,
                        track_id="salient-1",
                    )
                )

        return detections

    def _detect_person(self, small_rgb: Image.Image) -> Detection | None:
        ycbcr = np.asarray(small_rgb.convert("YCbCr"), dtype=np.float32)
        y = ycbcr[:, :, 0]
        cb = ycbcr[:, :, 1]
        cr = ycbcr[:, :, 2]

        skin_mask = (y > 55) & (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
        skin_ratio = float(skin_mask.mean())
        if skin_ratio < 0.015 or skin_ratio > 0.45:
            return None

        ys, xs = np.where(skin_mask)
        med_y, med_x = float(np.median(ys)), float(np.median(xs))
        keep = (np.abs(ys - med_y) < 16) & (np.abs(xs - med_x) < 16)
        if int(keep.sum()) < 22:
            return None

        f_ys, f_xs = ys[keep], xs[keep]
        fx1 = max(0.0, float(f_xs.min()) / 64.0 - 0.08)
        fy1 = max(0.0, float(f_ys.min()) / 48.0 - 0.06)
        fx2 = min(1.0, float(f_xs.max() + 1) / 64.0 + 0.08)
        fy2 = min(1.0, float(f_ys.max() + 1) / 48.0 + 0.35)

        return Detection(
            label="person",
            box=BoundingBox(
                x1=round(fx1, 3),
                y1=round(fy1, 3),
                x2=round(fx2, 3),
                y2=round(fy2, 3),
            ),
            confidence=0.84,
            model_version=self.model_version,
            track_id="person-1",
        )

    @staticmethod
    def _classify_salient_region(mean_rgb: Any, width: float, height: float) -> str:
        r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
        luma = 0.299 * r + 0.587 * g + 0.114 * b

        if b > r + 0.12 and b > g + 0.04:
            return "blue bottle" if height > width else "blue object"
        if r > 0.68 and g > 0.68 and b > 0.65:
            return "bright paper or screen" if width >= height else "white object"
        if r > g + 0.14 and r > b + 0.14:
            return "reddish object"
        if g > r + 0.10 and g > b + 0.10:
            return "greenish object"
        if width > 0.62 and height < 0.35:
            return "tabletop or surface"
        if luma < 0.25:
            return "dark object"
        return "foreground object"


DEFAULT_ONNX_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "weights" / "ssd_mobilenet_v1_12.onnx"
)


class OnnxCocoObjectDetector:
    """
    Pre-trained 80-class COCO object detector powered by ONNX Runtime CPU.

    Detects and draws individual bounding boxes around people, phones, bottles,
    cups, laptops, books, chairs, keyboards, etc. Falls back cleanly to
    HeuristicObjectDetector if onnxruntime or the model weights are unavailable.
    """

    model_version = "ssd-mobilenet-v1-coco-onnx"

    def __init__(
        self,
        model_path: Path | None = None,
        score_threshold: float = 0.42,
    ) -> None:
        self.model_path = model_path or DEFAULT_ONNX_MODEL_PATH
        self.score_threshold = score_threshold
        self._session: Any = None
        self._input_name: str = "inputs"
        self._fallback = HeuristicObjectDetector()
        self._init_session()

    def _init_session(self) -> None:
        if not self.model_path.exists():
            return
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]

            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
        except Exception:
            self._session = None

    def detect(self, frame: CameraFrame) -> list[Detection]:
        if self._session is None:
            return self._fallback.detect(frame)

        img = Image.open(io.BytesIO(frame.jpeg_bytes)).convert("RGB")
        # Pass 1: full frame
        all_dets = self._run_on_crop(img, BoundingBox(0.0, 0.0, 1.0, 1.0), tag="full")

        # Pass 2: center zoom crop (improves recall for smaller handheld/tabletop items)
        center_box = BoundingBox(0.18, 0.18, 0.82, 0.82)
        cw = int((center_box.x2 - center_box.x1) * img.width)
        ch = int((center_box.y2 - center_box.y1) * img.height)
        if cw >= 64 and ch >= 64:
            crop = img.crop(
                (
                    int(center_box.x1 * img.width),
                    int(center_box.y1 * img.height),
                    int(center_box.x2 * img.width),
                    int(center_box.y2 * img.height),
                )
            )
            all_dets.extend(self._run_on_crop(crop, center_box, tag="zoom"))

        deduped = self._nms_detections(all_dets, iou_threshold=0.50)
        if not deduped:
            return self._fallback.detect(frame)
        return deduped

    def _run_on_crop(self, crop: Image.Image, region: BoundingBox, tag: str) -> list[Detection]:
        resized = crop.resize((320, 240), Image.Resampling.BILINEAR)
        arr = np.expand_dims(np.asarray(resized, dtype=np.uint8), axis=0)

        boxes, classes, scores, num_det = self._session.run(None, {self._input_name: arr})
        count = int(num_det[0]) if len(num_det) else len(scores[0])

        rw = region.x2 - region.x1
        rh = region.y2 - region.y1
        detections: list[Detection] = []
        for i in range(min(count, len(scores[0]))):
            score = float(scores[0][i])
            if score < self.score_threshold:
                continue
            cls_id = int(classes[0][i])
            label = MSCOCO_90_LABELS.get(cls_id, f"object-{cls_id}")
            ymin, xmin, ymax, xmax = [float(v) for v in boxes[0][i]]
            x1 = max(0.0, min(1.0, region.x1 + xmin * rw))
            y1 = max(0.0, min(1.0, region.y1 + ymin * rh))
            x2 = max(x1 + 0.01, min(1.0, region.x1 + xmax * rw))
            y2 = max(y1 + 0.01, min(1.0, region.y1 + ymax * rh))
            detections.append(
                Detection(
                    label=label,
                    box=BoundingBox(
                        x1=round(x1, 3),
                        y1=round(y1, 3),
                        x2=round(x2, 3),
                        y2=round(y2, 3),
                    ),
                    confidence=round(score, 3),
                    model_version=self.model_version,
                    track_id=f"{tag}-{label}-{i}",
                )
            )
        return detections

    @staticmethod
    def _nms_detections(detections: list[Detection], iou_threshold: float = 0.50) -> list[Detection]:
        ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []
        for cand in ordered:
            duplicate = False
            for prev in kept:
                if prev.label != cand.label:
                    continue
                ix1 = max(prev.box.x1, cand.box.x1)
                iy1 = max(prev.box.y1, cand.box.y1)
                ix2 = min(prev.box.x2, cand.box.x2)
                iy2 = min(prev.box.y2, cand.box.y2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = prev.box.area + cand.box.area - inter
                if union > 1e-8 and (inter / union) >= iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(cand)
        return kept

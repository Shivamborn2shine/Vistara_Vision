from __future__ import annotations

import io
from typing import Any

from PIL import Image

from packages.vision.contracts import CameraFrame, OCRLine, OCRResult
from packages.vision.detector import assess_frame_lighting


class LocalOCREngine:
    """
    Pluggable OCR engine.

    Uses `pytesseract` or `rapidocr_onnxruntime` if installed, and cleanly abstains
    with helpful framing guidance when no OCR backend is installed or text is unreadable.
    """

    engine_version = "pluggable-ocr-v1"

    def __init__(self) -> None:
        self._backend: str = self._detect_backend()
        self._rapidocr: Any = None

    @staticmethod
    def _detect_backend() -> str:
        try:
            import rapidocr_onnxruntime  # noqa: F401

            return "rapidocr"
        except ImportError:
            pass
        try:
            import pytesseract  # noqa: F401

            return "tesseract"
        except ImportError:
            return "none"

    def read_text(self, frame: CameraFrame) -> OCRResult:
        _, lighting_issue = assess_frame_lighting(frame)
        if lighting_issue is not None:
            return OCRResult(
                full_text="",
                lines=[],
                confidence=0.0,
                engine_version=self.engine_version,
                abstained=True,
                guidance=lighting_issue,
            )

        if self._backend == "rapidocr":
            return self._read_with_rapidocr(frame)
        if self._backend == "tesseract":
            return self._read_with_tesseract(frame)

        return OCRResult(
            full_text="",
            lines=[],
            confidence=0.0,
            engine_version=self.engine_version,
            abstained=True,
            guidance=(
                "No local OCR engine (RapidOCR or Tesseract) is installed yet. "
                "Install rapidocr-onnxruntime to enable printed text reading."
            ),
        )

    def _read_with_rapidocr(self, frame: CameraFrame) -> OCRResult:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]

        if self._rapidocr is None:
            self._rapidocr = RapidOCR()

        img = Image.open(io.BytesIO(frame.jpeg_bytes)).convert("RGB")
        result, _ = self._rapidocr(np.asarray(img))
        if not result:
            return OCRResult(
                full_text="",
                lines=[],
                confidence=0.0,
                engine_version=f"{self.engine_version}:rapidocr",
                abstained=True,
                guidance="No readable printed text was found. Hold the page steady and flatter.",
            )

        lines: list[OCRLine] = []
        for item in result:
            text = str(item[1]).strip()
            conf = float(item[2])
            if text:
                lines.append(OCRLine(text=text, confidence=round(conf, 3)))

        full_text = " ".join(line.text for line in lines)
        avg_conf = sum(l.confidence for l in lines) / len(lines) if lines else 0.0
        return OCRResult(
            full_text=full_text,
            lines=lines,
            confidence=round(avg_conf, 3),
            engine_version=f"{self.engine_version}:rapidocr",
            abstained=not bool(lines),
            guidance=None if lines else "No readable text found.",
        )

    def _read_with_tesseract(self, frame: CameraFrame) -> OCRResult:
        import pytesseract  # type: ignore[import-not-found]

        img = Image.open(io.BytesIO(frame.jpeg_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img).strip()
        if not text:
            return OCRResult(
                full_text="",
                lines=[],
                confidence=0.0,
                engine_version=f"{self.engine_version}:tesseract",
                abstained=True,
                guidance="No readable printed text was found. Hold the page steady in better light.",
            )
        lines = [OCRLine(text=ln.strip(), confidence=0.8) for ln in text.splitlines() if ln.strip()]
        return OCRResult(
            full_text=" ".join(l.text for l in lines),
            lines=lines,
            confidence=0.8,
            engine_version=f"{self.engine_version}:tesseract",
            abstained=False,
        )

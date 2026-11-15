from __future__ import annotations

import io
from datetime import UTC, datetime

from PIL import Image

from packages.vision.contracts import BoundingBox, CameraFrame, CurrencyNoteCandidate
from packages.vision.currency import INRCurrencyAnalyzer


def _make_frame(rgb: tuple[int, int, int]) -> CameraFrame:
    img = Image.new("RGB", (160, 120), color=rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return CameraFrame(
        source_id="mock-cam-0",
        timestamp_utc=datetime.now(UTC),
        width=160,
        height=120,
        jpeg_bytes=buf.getvalue(),
    )


def test_currency_abstains_on_dark_frame() -> None:
    analyzer = INRCurrencyAnalyzer()
    dark_frame = _make_frame((5, 5, 5))
    res = analyzer.analyze([dark_frame])
    assert res.abstained is True
    assert res.uncertainty_reason == "poor_lighting"
    assert res.total_inr == 0
    assert "cannot verify whether a banknote is genuine" in res.authenticity_disclaimer.lower()


def test_currency_abstains_on_low_confidence_candidate() -> None:
    analyzer = INRCurrencyAnalyzer(confidence_threshold=0.75)
    frame = _make_frame((120, 120, 120))
    low_conf = CurrencyNoteCandidate(
        denomination_inr=500,
        confidence=0.48,
        box=BoundingBox(0.2, 0.2, 0.8, 0.8),
        track_id="note-1",
    )
    res = analyzer.analyze([frame], injected_candidates=[low_conf])
    assert res.abstained is True
    assert res.uncertainty_reason == "low_confidence"
    assert res.distinct_count == 0


def test_currency_deduplicates_multi_frame_note_and_counts_distinct() -> None:
    analyzer = INRCurrencyAnalyzer(confidence_threshold=0.70)
    frame = _make_frame((140, 130, 120))
    # Same note seen in two frames (same track_id and overlapping box) + a second distinct note
    candidates = [
        CurrencyNoteCandidate(500, 0.82, BoundingBox(0.1, 0.1, 0.45, 0.5), track_id="t1"),
        CurrencyNoteCandidate(500, 0.91, BoundingBox(0.11, 0.1, 0.46, 0.5), track_id="t1"),
        CurrencyNoteCandidate(100, 0.86, BoundingBox(0.55, 0.5, 0.9, 0.9), track_id="t2"),
    ]
    res = analyzer.analyze([frame], injected_candidates=candidates)
    assert res.abstained is False
    assert res.distinct_count == 2
    assert res.total_inr == 600
    assert "cannot verify whether a banknote is genuine" in res.authenticity_disclaimer.lower()

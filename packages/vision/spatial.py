from __future__ import annotations

import colorsys
import io

import numpy as np
from PIL import Image

from packages.vision.coco_labels import OBJECT_KNOWLEDGE_HINTS
from packages.vision.contracts import BoundingBox, CameraFrame, Detection
from packages.vision.detector import describe_relative_position


def describe_dominant_color(frame: CameraFrame, box: BoundingBox | None = None) -> str:
    """Estimate a human-friendly color name for the center region or bounding box."""
    img = Image.open(io.BytesIO(frame.jpeg_bytes)).convert("RGB")
    w, h = img.size
    b = box or BoundingBox(0.30, 0.30, 0.70, 0.70)
    crop = img.crop(
        (
            max(0, int(b.x1 * w)),
            max(0, int(b.y1 * h)),
            min(w, int(b.x2 * w)),
            min(h, int(b.y2 * h)),
        )
    )
    if crop.width < 4 or crop.height < 4:
        return "unknown color"

    arr = np.asarray(crop.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    med = np.median(arr, axis=(0, 1))
    r, g, b_val = float(med[0]), float(med[1]), float(med[2])
    hue, sat, val = colorsys.rgb_to_hsv(r, g, b_val)
    deg = hue * 360.0

    if val < 0.18:
        return "black or very dark"
    if sat < 0.12:
        if val > 0.78:
            return "white"
        return "grey"
    if deg < 15 or deg >= 340:
        return "red"
    if deg < 42:
        return "orange or brown"
    if deg < 68:
        return "yellow"
    if deg < 160:
        return "green"
    if deg < 205:
        return "cyan or light blue"
    if deg < 255:
        return "blue"
    if deg < 300:
        return "purple or lavender"
    return "pink or magenta"


def _spatial_relation(subject: Detection, anchor: Detection) -> str | None:
    """Compute a qualitative spatial relation between two detections."""
    sx, sy = subject.box.center_x, subject.box.center_y
    ax, ay = anchor.box.center_x, anchor.box.center_y

    # Check if subject overlaps inside/in front of a person
    if anchor.label == "person":
        if (
            anchor.box.x1 <= sx <= anchor.box.x2
            and anchor.box.y1 <= sy <= anchor.box.y2
            and subject.box.area < anchor.box.area * 0.85
        ):
            return "held or in front of the person"

    dx = sx - ax
    dy = sy - ay
    if abs(dx) > abs(dy) and abs(dx) > 0.15:
        return "to the right of the " + anchor.label if dx > 0 else "to the left of the " + anchor.label
    if abs(dy) > 0.18:
        return "below the " + anchor.label if dy > 0 else "above the " + anchor.label
    return None


def narrate_scene(
    detections: list[Detection],
    lighting_warning: str | None = None,
) -> str:
    """Produce a natural, spatially-grounded assistive scene description."""
    if lighting_warning:
        return lighting_warning
    if not detections:
        return "I do not see any distinct objects clearly in front of the camera. Try panning slowly."

    # Sort by personal match first, then area/confidence
    ordered = sorted(
        detections,
        key=lambda d: (d.is_personal_match, d.confidence, d.box.area),
        reverse=True,
    )

    # Group counts for repeated identical labels (e.g., 2 chairs)
    phrases: list[str] = []
    anchor = ordered[0]
    for idx, d in enumerate(ordered[:5]):
        pos = describe_relative_position(d.box)
        rel = _spatial_relation(d, anchor) if idx > 0 else None
        loc = f"{rel}, {pos}" if rel else pos
        tag = "your " if d.is_personal_match else ""
        hint = OBJECT_KNOWLEDGE_HINTS.get(d.label, "")
        hint_part = f" — {hint}" if hint and idx == 0 else ""
        phrases.append(f"{tag}{d.label} ({loc}, {int(d.confidence * 100)}%){hint_part}")

    return "In front of you, I see: " + "; ".join(phrases) + "."

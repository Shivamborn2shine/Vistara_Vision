from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class CameraStatus(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    PERMISSION_DENIED = "permission_denied"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalized bounding box [0.0, 1.0] in image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(slots=True)
class CameraFrame:
    """Single captured frame with source metadata and UTC timestamp."""

    source_id: str
    timestamp_utc: datetime
    width: int
    height: int
    jpeg_bytes: bytes
    sequence_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CameraInfo:
    """Metadata for an available camera source."""

    source_id: str
    name: str
    source_type: str  # "laptop_webcam" | "phone_webrtc" | "mock"
    available: bool = True


@dataclass(frozen=True, slots=True)
class Detection:
    """Detected object in a frame."""

    label: str
    box: BoundingBox
    confidence: float
    model_version: str
    track_id: str | None = None
    object_id: str | None = None
    is_personal_match: bool = False


@dataclass(frozen=True, slots=True)
class CurrencyNoteCandidate:
    """Single detected Indian banknote candidate."""

    denomination_inr: int
    confidence: float
    box: BoundingBox
    track_id: str | None = None


@dataclass(frozen=True, slots=True)
class CurrencyAnalysisResult:
    """Currency recognition output with strict safety and abstention semantics."""

    notes: list[CurrencyNoteCandidate]
    distinct_count: int
    total_inr: int
    abstained: bool
    uncertainty_reason: str | None
    guidance: str
    authenticity_disclaimer: str = (
        "Visual denomination estimate only. Cannot verify whether a banknote is genuine."
    )


@dataclass(frozen=True, slots=True)
class OCRLine:
    text: str
    confidence: float
    box: BoundingBox | None = None


@dataclass(frozen=True, slots=True)
class OCRResult:
    full_text: str
    lines: list[OCRLine]
    confidence: float
    engine_version: str
    abstained: bool = False
    guidance: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    values: list[float]
    encoder_version: str


@dataclass(frozen=True, slots=True)
class StableObservation:
    """Confidence- and stability-gated observation ready for episodic memory."""

    label: str
    confidence: float
    camera_id: str
    timestamp_utc: datetime
    landmark_relation: str
    box: BoundingBox
    object_id: str | None = None
    model_version: str = "unknown"
    scene_summary: str | None = None


@runtime_checkable
class CameraSource(Protocol):
    source_id: str

    def start(self) -> CameraStatus: ...
    def get_frame(self) -> CameraFrame | None: ...
    def status(self) -> CameraStatus: ...
    def stop(self) -> CameraStatus: ...


@runtime_checkable
class ObjectDetector(Protocol):
    model_version: str

    def detect(self, frame: CameraFrame) -> list[Detection]: ...


@runtime_checkable
class CurrencyAnalyzer(Protocol):
    model_version: str

    def analyze(self, frames: list[CameraFrame]) -> CurrencyAnalysisResult: ...


@runtime_checkable
class OCREngine(Protocol):
    engine_version: str

    def read_text(self, frame: CameraFrame) -> OCRResult: ...


@runtime_checkable
class VisualEncoder(Protocol):
    encoder_version: str

    def encode(self, frame: CameraFrame, box: BoundingBox | None = None) -> EmbeddingVector: ...


def utc_now() -> datetime:
    return datetime.now(UTC)

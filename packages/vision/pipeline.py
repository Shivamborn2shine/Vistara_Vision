from __future__ import annotations

import queue
from dataclasses import dataclass

from packages.vision.contracts import (
    CameraFrame,
    Detection,
    ObjectDetector,
    StableObservation,
    VisualEncoder,
)
from packages.vision.detector import describe_relative_position
from packages.vision.encoder import cosine_similarity


@dataclass(frozen=True, slots=True)
class EnrolledObjectProfile:
    object_id: str
    label: str
    embeddings: list[list[float]]


@dataclass(frozen=True, slots=True)
class PersonalMatchResult:
    object_id: str
    label: str
    similarity: float
    needs_confirmation: bool


class BoundedFrameQueue:
    """Bounded queue that drops oldest video frames under load; never blocks callers."""

    def __init__(self, maxsize: int = 2) -> None:
        self._queue: queue.Queue[CameraFrame] = queue.Queue(maxsize=maxsize)
        self.dropped_frames: int = 0

    def push(self, frame: CameraFrame) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.dropped_frames += 1

    def pop(self) -> CameraFrame | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def clear(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


class PerceptionPipeline:
    """
    Executes CameraFrame -> Detections -> Personal Matching -> StableObservation.

    Enforces confidence and stability thresholds so transient false positives do not
    pollute episodic memory.
    """

    def __init__(
        self,
        detector: ObjectDetector,
        encoder: VisualEncoder,
        min_confidence: float = 0.60,
        personal_match_threshold: float = 0.86,
        personal_confirm_threshold: float = 0.76,
    ) -> None:
        self.detector = detector
        self.encoder = encoder
        self.min_confidence = min_confidence
        self.personal_match_threshold = personal_match_threshold
        self.personal_confirm_threshold = personal_confirm_threshold
        self._stability_counts: dict[str, int] = {}

    def match_personal_objects(
        self,
        frame: CameraFrame,
        profiles: list[EnrolledObjectProfile],
        detection: Detection | None = None,
    ) -> PersonalMatchResult | None:
        if not profiles:
            return None
        emb = self.encoder.encode(frame, box=detection.box if detection else None)
        best_profile: EnrolledObjectProfile | None = None
        best_sim = -1.0

        for prof in profiles:
            for ref_vec in prof.embeddings:
                sim = cosine_similarity(emb.values, ref_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_profile = prof

        if best_profile is None or best_sim < self.personal_confirm_threshold:
            return None

        return PersonalMatchResult(
            object_id=best_profile.object_id,
            label=best_profile.label,
            similarity=round(best_sim, 3),
            needs_confirmation=best_sim < self.personal_match_threshold,
        )

    def process_frame(
        self,
        frame: CameraFrame,
        profiles: list[EnrolledObjectProfile] | None = None,
        require_stability: int = 1,
    ) -> tuple[list[Detection], list[StableObservation]]:
        raw_detections = self.detector.detect(frame)
        enriched: list[Detection] = []
        stable: list[StableObservation] = []

        for det in raw_detections:
            if det.confidence < self.min_confidence:
                continue

            personal = self.match_personal_objects(frame, profiles or [], det)
            if personal and not personal.needs_confirmation:
                det = Detection(
                    label=personal.label,
                    box=det.box,
                    confidence=max(det.confidence, personal.similarity),
                    model_version=det.model_version,
                    track_id=det.track_id,
                    object_id=personal.object_id,
                    is_personal_match=True,
                )
            enriched.append(det)

            key = f"{frame.source_id}:{det.object_id or det.label}"
            self._stability_counts[key] = self._stability_counts.get(key, 0) + 1
            if self._stability_counts[key] >= require_stability:
                stable.append(
                    StableObservation(
                        label=det.label,
                        confidence=det.confidence,
                        camera_id=frame.source_id,
                        timestamp_utc=frame.timestamp_utc,
                        landmark_relation=describe_relative_position(det.box),
                        box=det.box,
                        object_id=det.object_id,
                        model_version=det.model_version,
                    )
                )

        return enriched, stable

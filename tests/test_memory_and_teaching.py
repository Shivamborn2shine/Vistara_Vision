from __future__ import annotations

from datetime import UTC, datetime, timedelta

from packages.memory.answering import MemoryAnswerEngine
from packages.memory.repository import MemoryRepository
from packages.vision.contracts import BoundingBox, StableObservation


def test_teach_update_and_delete_personal_object() -> None:
    repo = MemoryRepository(":memory:")
    rec = repo.enroll_object(
        label="Blue Water Bottle",
        embeddings=[([0.1, 0.2, 0.3], "local-hist-v1")],
        aliases=["bottle", "flask"],
        owner_note="Metal bottle with blue cap",
    )
    assert rec.label == "Blue Water Bottle"
    assert rec.embedding_count == 1

    updated = repo.update_object(rec.object_id, label="My Blue Bottle")
    assert updated is not None
    assert updated.label == "My Blue Bottle"

    assert len(repo.list_objects()) == 1
    assert repo.delete_object(rec.object_id) is True
    assert len(repo.list_objects()) == 0


def test_honest_last_seen_recall_and_conflict_warning() -> None:
    repo = MemoryRepository(":memory:")
    engine = MemoryAnswerEngine(repo)
    now = datetime.now(UTC)

    # Unseen item
    miss = engine.answer_last_seen("keys", now=now)
    assert miss.found is False
    assert "not observed" in miss.spoken_answer.lower()

    # Two observations in different places
    repo.write_observation(
        StableObservation(
            label="keys",
            confidence=0.88,
            camera_id="webcam-0",
            timestamp_utc=now - timedelta(minutes=30),
            landmark_relation="on the left, middle area",
            box=BoundingBox(0.1, 0.4, 0.2, 0.5),
        ),
        label_source="model_inferred",
    )
    repo.write_observation(
        StableObservation(
            label="keys",
            confidence=0.92,
            camera_id="webcam-0",
            timestamp_utc=now - timedelta(minutes=5),
            landmark_relation="on the right, lower area",
            box=BoundingBox(0.7, 0.7, 0.8, 0.8),
        ),
        label_source="user_confirmed",
    )

    ans = engine.answer_last_seen("keys", now=now)
    assert ans.found is True
    assert ans.conflicting_sightings == 1
    assert "5 minutes ago" in ans.spoken_answer
    assert "on the right, lower area" in ans.spoken_answer
    assert "confirmed by you" in ans.spoken_answer
    assert "may have been moved" in ans.spoken_answer.lower()

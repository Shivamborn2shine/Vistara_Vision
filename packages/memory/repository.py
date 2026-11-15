from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine, delete, desc, select
from sqlalchemy.orm import Session, sessionmaker

from packages.memory.models import (
    Base,
    ObjectEmbedding,
    Observation,
    PersonalObject,
    UserFeedback,
    VisionSession,
)
from packages.vision.contracts import StableObservation
from packages.vision.pipeline import EnrolledObjectProfile


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    event_id: str
    object_id: str | None
    label: str
    label_source: str
    timestamp_utc: datetime
    camera_id: str
    room_id: str | None
    landmark_relation: str
    confidence: float


@dataclass(frozen=True, slots=True)
class PersonalObjectRecord:
    object_id: str
    label: str
    aliases: list[str]
    owner_note: str | None
    embedding_count: int
    created_at: datetime
    edited_at: datetime


class MemoryRepository:
    """SQLite-backed repository for teachable objects, observations, feedback, and sessions."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        url = (
            "sqlite:///:memory:"
            if str(db_path) == ":memory:"
            else f"sqlite:///{Path(db_path).as_posix()}"
        )
        self.engine = create_engine(url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def _session(self) -> Session:
        return self._session_factory()

    # --- Teachable Objects (Semantic Memory) ---

    def enroll_object(
        self,
        label: str,
        embeddings: list[tuple[list[float], str]],
        aliases: list[str] | None = None,
        owner_note: str | None = None,
    ) -> PersonalObjectRecord:
        clean_label = label.strip()
        if not clean_label:
            raise ValueError("Object label cannot be empty.")

        now = datetime.now(UTC)
        with self._session() as db:
            obj = PersonalObject(
                label=clean_label,
                aliases_json=json.dumps(aliases or []),
                owner_note=owner_note,
                created_at=now,
                edited_at=now,
            )
            db.add(obj)
            db.flush()
            for vec, enc_ver in embeddings:
                db.add(
                    ObjectEmbedding(
                        object_id=obj.object_id,
                        encoder_version=enc_ver,
                        vector_json=json.dumps(vec),
                        created_at=now,
                    )
                )
            db.commit()
            return PersonalObjectRecord(
                object_id=obj.object_id,
                label=obj.label,
                aliases=aliases or [],
                owner_note=obj.owner_note,
                embedding_count=len(embeddings),
                created_at=obj.created_at,
                edited_at=obj.edited_at,
            )

    def add_embedding(self, object_id: str, vector: list[float], encoder_version: str) -> bool:
        with self._session() as db:
            obj = db.get(PersonalObject, object_id)
            if obj is None:
                return False
            db.add(
                ObjectEmbedding(
                    object_id=object_id,
                    encoder_version=encoder_version,
                    vector_json=json.dumps(vector),
                )
            )
            obj.edited_at = datetime.now(UTC)
            db.commit()
            return True

    def list_objects(self) -> list[PersonalObjectRecord]:
        with self._session() as db:
            objs = db.scalars(select(PersonalObject).order_by(PersonalObject.label)).all()
            return [
                PersonalObjectRecord(
                    object_id=o.object_id,
                    label=o.label,
                    aliases=json.loads(o.aliases_json),
                    owner_note=o.owner_note,
                    embedding_count=len(o.embeddings),
                    created_at=o.created_at,
                    edited_at=o.edited_at,
                )
                for o in objs
            ]

    def load_enrolled_profiles(self) -> list[EnrolledObjectProfile]:
        with self._session() as db:
            objs = db.scalars(select(PersonalObject)).all()
            return [
                EnrolledObjectProfile(
                    object_id=o.object_id,
                    label=o.label,
                    embeddings=[json.loads(e.vector_json) for e in o.embeddings],
                )
                for o in objs
                if o.embeddings
            ]

    def update_object(
        self,
        object_id: str,
        label: str | None = None,
        aliases: list[str] | None = None,
        owner_note: str | None = None,
    ) -> PersonalObjectRecord | None:
        with self._session() as db:
            obj = db.get(PersonalObject, object_id)
            if obj is None:
                return None
            if label is not None and label.strip():
                obj.label = label.strip()
            if aliases is not None:
                obj.aliases_json = json.dumps(aliases)
            if owner_note is not None:
                obj.owner_note = owner_note
            obj.edited_at = datetime.now(UTC)
            db.commit()
            return PersonalObjectRecord(
                object_id=obj.object_id,
                label=obj.label,
                aliases=json.loads(obj.aliases_json),
                owner_note=obj.owner_note,
                embedding_count=len(obj.embeddings),
                created_at=obj.created_at,
                edited_at=obj.edited_at,
            )

    def delete_object(self, object_id: str) -> bool:
        with self._session() as db:
            obj = db.get(PersonalObject, object_id)
            if obj is None:
                return False
            db.delete(obj)
            db.commit()
            return True

    # --- Episodic Memory (Observations) ---

    def write_observation(
        self,
        obs: StableObservation,
        label_source: str = "model_inferred",
        room_id: str | None = None,
        snapshot_ref: str | None = None,
    ) -> ObservationRecord:
        ts = (
            obs.timestamp_utc if obs.timestamp_utc.tzinfo else obs.timestamp_utc.replace(tzinfo=UTC)
        )
        with self._session() as db:
            row = Observation(
                object_id=obs.object_id,
                label=obs.label,
                label_source=label_source,
                timestamp_utc=ts,
                camera_id=obs.camera_id,
                room_id=room_id,
                landmark_relation=obs.landmark_relation,
                bbox_json=json.dumps(
                    {"x1": obs.box.x1, "y1": obs.box.y1, "x2": obs.box.x2, "y2": obs.box.y2}
                ),
                confidence=float(obs.confidence),
                snapshot_ref=snapshot_ref,
            )
            db.add(row)
            db.commit()
            return self._to_obs_record(row)

    def search_last_seen(self, query: str, limit: int = 5) -> list[ObservationRecord]:
        term = query.strip().lower()
        if not term:
            return []
        with self._session() as db:
            # Match personal object label/aliases or observation label
            rows = db.scalars(
                select(Observation).order_by(desc(Observation.timestamp_utc)).limit(200)
            ).all()
            matched: list[ObservationRecord] = []
            for r in rows:
                if term in r.label.lower():
                    matched.append(self._to_obs_record(r))
                elif r.personal_object is not None:
                    aliases = [a.lower() for a in json.loads(r.personal_object.aliases_json)]
                    if term in r.personal_object.label.lower() or any(term in a for a in aliases):
                        matched.append(self._to_obs_record(r))
                if len(matched) >= limit:
                    break
            return matched

    def list_recent_observations(self, limit: int = 20) -> list[ObservationRecord]:
        with self._session() as db:
            rows = db.scalars(
                select(Observation).order_by(desc(Observation.timestamp_utc)).limit(limit)
            ).all()
            return [self._to_obs_record(r) for r in rows]

    def correct_observation(self, event_id: str, corrected_label: str) -> ObservationRecord | None:
        clean = corrected_label.strip()
        if not clean:
            return None
        with self._session() as db:
            row = db.get(Observation, event_id)
            if row is None:
                return None
            prev_label = row.label
            row.label = clean
            row.label_source = "user_confirmed"
            db.add(
                UserFeedback(
                    prediction=prev_label,
                    user_correction=clean,
                    affected_observation_id=event_id,
                )
            )
            db.commit()
            return self._to_obs_record(row)

    def delete_observation(self, event_id: str) -> bool:
        with self._session() as db:
            row = db.get(Observation, event_id)
            if row is None:
                return False
            db.delete(row)
            db.commit()
            return True

    def clear_all_memory(self) -> None:
        with self._session() as db:
            db.execute(delete(Observation))
            db.execute(delete(UserFeedback))
            db.execute(delete(ObjectEmbedding))
            db.execute(delete(PersonalObject))
            db.execute(delete(VisionSession))
            db.commit()

    def export_memory(self) -> dict[str, object]:
        return {
            "objects": [
                {
                    "object_id": o.object_id,
                    "label": o.label,
                    "aliases": o.aliases,
                    "owner_note": o.owner_note,
                    "embedding_count": o.embedding_count,
                }
                for o in self.list_objects()
            ],
            "observations": [
                {
                    "event_id": r.event_id,
                    "object_id": r.object_id,
                    "label": r.label,
                    "label_source": r.label_source,
                    "timestamp_utc": r.timestamp_utc.isoformat(),
                    "camera_id": r.camera_id,
                    "room_id": r.room_id,
                    "landmark_relation": r.landmark_relation,
                    "confidence": r.confidence,
                }
                for r in self.list_recent_observations(limit=500)
            ],
        }

    # --- Sessions ---

    def start_session(self, source_type: str, permission_state: str = "granted") -> str:
        with self._session() as db:
            sess = VisionSession(source_type=source_type, permission_state=permission_state)
            db.add(sess)
            db.commit()
            return sess.session_id

    def end_session(self, session_id: str) -> None:
        with self._session() as db:
            sess = db.get(VisionSession, session_id)
            if sess is not None:
                sess.ended_at = datetime.now(UTC)
                db.commit()

    @staticmethod
    def _to_obs_record(row: Observation) -> ObservationRecord:
        ts = (
            row.timestamp_utc if row.timestamp_utc.tzinfo else row.timestamp_utc.replace(tzinfo=UTC)
        )
        return ObservationRecord(
            event_id=row.event_id,
            object_id=row.object_id,
            label=row.label,
            label_source=row.label_source,
            timestamp_utc=ts,
            camera_id=row.camera_id,
            room_id=row.room_id,
            landmark_relation=row.landmark_relation,
            confidence=row.confidence,
        )

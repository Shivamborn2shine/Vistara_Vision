from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class PersonalObject(Base):
    __tablename__ = "personal_objects"

    object_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    label: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    aliases_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    owner_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    edited_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    embeddings: Mapped[list[ObjectEmbedding]] = relationship(
        back_populates="personal_object", cascade="all, delete-orphan"
    )
    observations: Mapped[list[Observation]] = relationship(back_populates="personal_object")


class ObjectEmbedding(Base):
    __tablename__ = "object_embeddings"

    embedding_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    object_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("personal_objects.object_id", ondelete="CASCADE"), index=True
    )
    encoder_version: Mapped[str] = mapped_column(String(64), nullable=False)
    vector_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    personal_object: Mapped[PersonalObject] = relationship(back_populates="embeddings")


class Observation(Base):
    __tablename__ = "observations"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    object_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("personal_objects.object_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    label: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    label_source: Mapped[str] = mapped_column(String(32), nullable=False, default="model_inferred")
    timestamp_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, index=True
    )
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    room_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    landmark_relation: Mapped[str] = mapped_column(String(255), nullable=False)
    bbox_json: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    snapshot_ref: Mapped[str | None] = mapped_column(Text, nullable=True)

    personal_object: Mapped[PersonalObject | None] = relationship(back_populates="observations")


class VisionSession(Base):
    __tablename__ = "vision_sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    permission_state: Mapped[str] = mapped_column(String(32), nullable=False, default="granted")


class UserFeedback(Base):
    __tablename__ = "user_feedback"

    feedback_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    prediction: Mapped[str] = mapped_column(String(120), nullable=False)
    user_correction: Mapped[str] = mapped_column(String(120), nullable=False)
    affected_observation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

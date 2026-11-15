from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from packages.memory.repository import MemoryRepository, ObservationRecord


def format_elapsed_time(observed_at: datetime, now: datetime | None = None) -> str:
    ref = now or datetime.now(UTC)
    obs_utc = observed_at if observed_at.tzinfo else observed_at.replace(tzinfo=UTC)
    delta = max(0, int((ref - obs_utc).total_seconds()))
    if delta < 60:
        return f"{delta} seconds ago"
    minutes = delta // 60
    if minutes < 60:
        return f"{minutes} {'minute' if minutes == 1 else 'minutes'} ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} {'hour' if hours == 1 else 'hours'} ago"
    days = hours // 24
    return f"{days} {'day' if days == 1 else 'days'} ago"


@dataclass(frozen=True, slots=True)
class LastSeenAnswer:
    found: bool
    spoken_answer: str
    latest_observation: ObservationRecord | None
    conflicting_sightings: int


class MemoryAnswerEngine:
    """
    Answers 'Where was my X?' strictly from stored observations.

    Never claims live presence from memory alone, reports elapsed time and
    relative landmark context, and flags conflicting sightings if observed in
    different places recently.
    """

    def __init__(self, repo: MemoryRepository) -> None:
        self.repo = repo

    def answer_last_seen(self, item_query: str, now: datetime | None = None) -> LastSeenAnswer:
        matches = self.repo.search_last_seen(item_query, limit=5)
        if not matches:
            return LastSeenAnswer(
                found=False,
                spoken_answer=(
                    f"I have not observed '{item_query}' in your saved memory yet. "
                    "If it is nearby, start the camera so I can look around."
                ),
                latest_observation=None,
                conflicting_sightings=0,
            )

        latest = matches[0]
        elapsed = format_elapsed_time(latest.timestamp_utc, now=now)
        room_part = f" in {latest.room_id}" if latest.room_id else ""
        source_note = (
            "confirmed by you"
            if latest.label_source == "user_confirmed"
            else f"detected with {int(latest.confidence * 100)}% confidence"
        )

        # Check if earlier recent sightings had a different landmark relation
        distinct_places = {m.landmark_relation for m in matches}
        conflicts = max(0, len(distinct_places) - 1)
        conflict_note = ""
        if conflicts > 0:
            conflict_note = (
                " Note: I also saw it in another position earlier, "
                "and it may have been moved since then."
            )
        else:
            conflict_note = " I have not observed it since, so it may have been moved."

        spoken = (
            f"Your {latest.label} was last observed {elapsed}{room_part}, "
            f"{latest.landmark_relation} (via {latest.camera_id}, {source_note})."
            f"{conflict_note}"
        )
        return LastSeenAnswer(
            found=True,
            spoken_answer=spoken,
            latest_observation=latest,
            conflicting_sightings=conflicts,
        )

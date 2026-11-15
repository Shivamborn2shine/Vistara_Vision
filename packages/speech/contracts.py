from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SpeechEngine(Protocol):
    def transcribe(self, audio_bytes: bytes) -> str: ...
    def speak(self, text: str, rate: float = 1.0) -> None: ...
    def cancel(self) -> None: ...


class TextFallbackSpeechEngine:
    """No-op local speech adapter for headless/text-only mode and tests."""

    def __init__(self) -> None:
        self.spoken_history: list[str] = []

    def transcribe(self, audio_bytes: bytes) -> str:
        return ""

    def speak(self, text: str, rate: float = 1.0) -> None:
        self.spoken_history.append(text)

    def cancel(self) -> None:
        pass

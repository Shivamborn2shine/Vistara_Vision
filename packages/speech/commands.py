from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class CommandIntent(StrEnum):
    START_VISION = "start_vision"
    STOP_CAMERA = "stop_camera"
    DESCRIBE_SCENE = "describe_scene"
    READ_TEXT = "read_text"
    COUNT_CURRENCY = "count_currency"
    TEACH_OBJECT = "teach_object"
    CORRECT_OBJECT = "correct_object"
    ASK_MEMORY = "ask_memory"
    PAIR_PHONE = "pair_phone"
    REPEAT = "repeat"
    STOP_SPEAKING = "stop_speaking"
    FORGET_LAST = "forget_last"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    intent: CommandIntent
    argument: str | None = None
    raw_text: str = ""


def parse_voice_command(text: str) -> ParsedCommand:
    """Parse spoken or typed assistive commands into structured intents."""
    raw = text.strip()
    low = raw.lower()
    low = re.sub(r"[?.!,]+$", "", low).strip()

    if not low:
        return ParsedCommand(intent=CommandIntent.UNKNOWN, raw_text=raw)

    if any(k in low for k in ("stop speaking", "be quiet", "cancel speech", "shut up")):
        return ParsedCommand(intent=CommandIntent.STOP_SPEAKING, raw_text=raw)

    if low in ("repeat", "say that again", "repeat that"):
        return ParsedCommand(intent=CommandIntent.REPEAT, raw_text=raw)

    if any(k in low for k in ("forget that", "forget last", "erase that observation")):
        return ParsedCommand(intent=CommandIntent.FORGET_LAST, raw_text=raw)

    if any(k in low for k in ("start vision", "start camera", "open camera", "turn on camera")):
        return ParsedCommand(intent=CommandIntent.START_VISION, raw_text=raw)

    if any(k in low for k in ("stop camera", "pause camera", "stop vision", "turn off camera")):
        return ParsedCommand(intent=CommandIntent.STOP_CAMERA, raw_text=raw)

    if any(k in low for k in ("connect phone", "pair phone", "phone camera")):
        return ParsedCommand(intent=CommandIntent.PAIR_PHONE, raw_text=raw)

    if any(
        k in low
        for k in (
            "what note",
            "which note",
            "count these notes",
            "count the notes",
            "count currency",
            "count cash",
        )
    ):
        return ParsedCommand(intent=CommandIntent.COUNT_CURRENCY, raw_text=raw)

    if any(k in low for k in ("read this", "read text", "read label", "what does it say")):
        return ParsedCommand(intent=CommandIntent.READ_TEXT, raw_text=raw)

    if any(
        k in low
        for k in ("what is in front of me", "what do you see", "describe scene", "what is here")
    ):
        return ParsedCommand(intent=CommandIntent.DESCRIBE_SCENE, raw_text=raw)

    # "No, that is my charger" / "That is my keys"
    m_corr = re.search(r"(?:no,?\s*)?(?:that is|that's|it is|it's)\s+(?:my\s+)?(.+)$", low)
    if m_corr:
        return ParsedCommand(
            intent=CommandIntent.CORRECT_OBJECT,
            argument=m_corr.group(1).strip(),
            raw_text=raw,
        )

    # "Learn this as my blue bottle" / "Learn this object"
    m_teach = re.search(r"learn\s+this(?:\s+object)?(?:\s+as\s+(?:my\s+)?)?(.+)?$", low)
    if m_teach:
        arg = (m_teach.group(1) or "").strip() or None
        return ParsedCommand(intent=CommandIntent.TEACH_OBJECT, argument=arg, raw_text=raw)

    # "Where was my bottle" / "Where did you last see my keys"
    m_mem = re.search(r"where\s+(?:was|is|did you last see)\s+(?:my\s+|the\s+)?(.+)$", low)
    if m_mem:
        return ParsedCommand(
            intent=CommandIntent.ASK_MEMORY,
            argument=m_mem.group(1).strip(),
            raw_text=raw,
        )

    return ParsedCommand(intent=CommandIntent.UNKNOWN, raw_text=raw)

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
    CHECK_PRESENCE = "check_presence"
    DESCRIBE_COLOR = "describe_color"
    LIST_KNOWN = "list_known"
    GET_TIME = "get_time"
    GREETING = "greeting"
    CHECK_LIGHTING = "check_lighting"
    HELP = "help"
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

    if low in (
        "hi",
        "hello",
        "hey",
        "hey vistara",
        "hi vistara",
        "hello vistara",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
    ):
        return ParsedCommand(intent=CommandIntent.GREETING, raw_text=raw)

    if any(
        k in low
        for k in (
            "what time",
            "what's the time",
            "what is the time",
            "tell me the time",
            "current time",
            "what date",
            "what's the date",
            "what is the date",
            "today's date",
            "what day is it",
        )
    ):
        return ParsedCommand(intent=CommandIntent.GET_TIME, raw_text=raw)

    if any(
        k in low
        for k in (
            "how is the light",
            "how's the light",
            "check lighting",
            "check light",
            "is it dark",
            "is it bright",
        )
    ):
        return ParsedCommand(intent=CommandIntent.CHECK_LIGHTING, raw_text=raw)

    if any(k in low for k in ("what color", "what colour", "which color", "which colour")):
        return ParsedCommand(intent=CommandIntent.DESCRIBE_COLOR, raw_text=raw)

    if any(
        k in low
        for k in (
            "what objects do you know",
            "list learned objects",
            "what do you remember",
            "what did you see earlier",
        )
    ):
        return ParsedCommand(intent=CommandIntent.LIST_KNOWN, raw_text=raw)

    if low in ("help", "what can you do", "commands", "voice commands", "what commands"):
        return ParsedCommand(intent=CommandIntent.HELP, raw_text=raw)

    # "Is there a bottle?" / "Do you see my keys?" / "How many people are there?"
    m_pres = re.search(
        r"(?:is there|are there|do you see|can you see|how many)\s+(?:a\s+|an\s+|any\s+|my\s+|the\s+)?(.+?)(?:\s+here|\s+in front of me|\s+around)?$",
        low,
    )
    if m_pres:
        return ParsedCommand(
            intent=CommandIntent.CHECK_PRESENCE,
            argument=m_pres.group(1).strip(),
            raw_text=raw,
        )

    if any(
        k in low
        for k in (
            "stop speaking",
            "stop talking",
            "be quiet",
            "cancel speech",
            "shut up",
            "silence",
            "quiet",
        )
    ):
        return ParsedCommand(intent=CommandIntent.STOP_SPEAKING, raw_text=raw)

    if low in ("repeat", "say that again", "repeat that", "pardon", "what did you say"):
        return ParsedCommand(intent=CommandIntent.REPEAT, raw_text=raw)

    if any(k in low for k in ("forget that", "forget last", "erase that observation")):
        return ParsedCommand(intent=CommandIntent.FORGET_LAST, raw_text=raw)

    if any(k in low for k in ("start vision", "start camera", "open camera", "turn on camera")):
        return ParsedCommand(intent=CommandIntent.START_VISION, raw_text=raw)

    if any(
        k in low
        for k in ("stop camera", "pause camera", "stop vision", "turn off camera", "close camera")
    ):
        return ParsedCommand(intent=CommandIntent.STOP_CAMERA, raw_text=raw)

    if any(k in low for k in ("connect phone", "pair phone", "phone camera", "use phone camera")):
        return ParsedCommand(intent=CommandIntent.PAIR_PHONE, raw_text=raw)

    if any(
        k in low
        for k in (
            "what note",
            "which note",
            "what is this note",
            "identify currency",
            "identify note",
            "recognize currency",
            "check currency",
            "check money",
            "how much money",
            "how much cash",
            "count these notes",
            "count the notes",
            "count currency",
            "count cash",
        )
    ):
        return ParsedCommand(intent=CommandIntent.COUNT_CURRENCY, raw_text=raw)

    if any(
        k in low
        for k in (
            "read this",
            "read text",
            "read the text",
            "read label",
            "what does it say",
            "read it",
            "read what is written",
            "scan text",
        )
    ):
        return ParsedCommand(intent=CommandIntent.READ_TEXT, raw_text=raw)

    if any(
        k in low
        for k in (
            "what is in front of me",
            "what's in front of me",
            "what do you see",
            "what can you see",
            "tell me what you see",
            "describe scene",
            "describe my surroundings",
            "look around",
            "what is here",
            "what is around me",
        )
    ) or low == "describe":
        return ParsedCommand(intent=CommandIntent.DESCRIBE_SCENE, raw_text=raw)

    # "No, that is my charger" / "That is my keys"
    m_corr = re.search(r"(?:no,?\s*)?(?:that is|that's|it is|it's)\s+(?:my\s+)?(.+)$", low)
    if m_corr:
        return ParsedCommand(
            intent=CommandIntent.CORRECT_OBJECT,
            argument=m_corr.group(1).strip(),
            raw_text=raw,
        )

    # "Learn this as my blue bottle" / "Remember this as my keys" / "Learn this object"
    m_teach = re.search(
        r"(?:learn|remember|save)\s+this(?:\s+object)?(?:\s+as\s+(?:my\s+)?)?(.+)?$", low
    )
    if m_teach:
        arg = (m_teach.group(1) or "").strip() or None
        return ParsedCommand(intent=CommandIntent.TEACH_OBJECT, argument=arg, raw_text=raw)

    # "Where was my bottle" / "Where did you last see my keys" / "Find my bottle"
    m_mem = re.search(
        r"(?:where\s+(?:was|is|did you last see)|find|did you see)\s+(?:my\s+|the\s+)?(.+)$", low
    )
    if m_mem:
        return ParsedCommand(
            intent=CommandIntent.ASK_MEMORY,
            argument=m_mem.group(1).strip(),
            raw_text=raw,
        )

    return ParsedCommand(intent=CommandIntent.UNKNOWN, raw_text=raw)

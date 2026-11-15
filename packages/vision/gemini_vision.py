from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from packages.vision.contracts import CameraFrame

_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_SYSTEM_PROMPT = (
    "You are Vistara Vision, a voice-first assistive AI for a blind or low-vision user. "
    "Answer the user's question clearly, accurately, and concisely in plain spoken English "
    "(2 to 4 short sentences unless they ask for detailed reading). "
    "When describing a camera image, mention key objects, text, colors, and relative positions "
    "(in front of you, to your left, to your right, closer, farther). "
    "If the image is blurry, dark, or does not clearly show what the user asked about, say so honestly. "
    "For banknotes, state the denomination if visible and note that you cannot verify authenticity."
)


@dataclass(frozen=True, slots=True)
class GeminiVisionReply:
    text: str
    model: str
    ok: bool
    error_reason: str | None = None


class GeminiVisionAssistant:
    """Multimodal Gemini Vision client for open-ended scene, text, and assistive Q&A."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.5-flash",
        config_path: Path | None = None,
        timeout_seconds: float = 25.0,
    ) -> None:
        self._api_key = (api_key or "").strip() or None
        self._model = model.strip() or "gemini-3.5-flash"
        if self._model == "gemini-2.5-flash":
            self._model = "gemini-3.5-flash"
        self._config_path = config_path
        self._timeout = timeout_seconds
        self._load_persisted_config()

    def _load_persisted_config(self) -> None:
        if self._config_path is None or not self._config_path.exists():
            return
        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            saved_key = str(data.get("api_key", "")).strip()
            saved_model = str(data.get("model", "")).strip()
            if not self._api_key and saved_key:
                self._api_key = saved_key
            if saved_model:
                self._model = "gemini-3.5-flash" if saved_model == "gemini-2.5-flash" else saved_model
        except Exception:
            pass

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    @property
    def model(self) -> str:
        return self._model

    def configure(self, api_key: str, model: str | None = None) -> None:
        cleaned_key = api_key.strip()
        self._api_key = cleaned_key or None
        if model and model.strip():
            self._model = model.strip()
        if self._config_path is not None:
            if self._api_key:
                self._config_path.write_text(
                    json.dumps({"api_key": self._api_key, "model": self._model}, indent=2),
                    encoding="utf-8",
                )
            elif self._config_path.exists():
                self._config_path.unlink()

    def ask(
        self,
        prompt: str,
        frame: CameraFrame | None = None,
        known_objects: list[str] | None = None,
    ) -> GeminiVisionReply:
        if not self._api_key:
            return GeminiVisionReply(
                text=(
                    "Gemini Vision is not configured yet. "
                    "Add your Gemini API key in Settings to ask general AI questions."
                ),
                model=self._model,
                ok=False,
                error_reason="not_configured",
            )

        user_parts: list[dict[str, Any]] = []
        if frame is not None and frame.jpeg_bytes:
            user_parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(frame.jpeg_bytes).decode("ascii"),
                    }
                }
            )

        context_lines: list[str] = []
        if known_objects:
            context_lines.append(
                "User's taught personal objects: " + ", ".join(known_objects[:15]) + "."
            )
        if frame is None:
            context_lines.append("Note: No live camera frame is currently active.")
        context_lines.append(f"User request: {prompt.strip()}")
        user_parts.append({"text": "\n".join(context_lines)})

        payload: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": user_parts}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 320,
            },
        }

        models_to_try: list[str] = [self._model]
        for fb in ("gemini-3.5-flash", "gemini-3.5-flash-lite"):
            if fb not in models_to_try:
                models_to_try.append(fb)

        resp: httpx.Response | None = None
        used_model = self._model
        last_exc: str | None = None

        for candidate_model in models_to_try:
            used_model = candidate_model
            url = _GEMINI_ENDPOINT.format(model=candidate_model)
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    resp = client.post(
                        url,
                        headers={"x-goog-api-key": self._api_key},
                        json=payload,
                    )
                last_exc = None
                if resp.status_code in (404, 503):
                    continue
                break
            except httpx.TimeoutException:
                last_exc = "timeout"
                resp = None
                continue
            except Exception:
                return GeminiVisionReply(
                    text="Could not reach Gemini Vision. Please check your internet connection.",
                    model=used_model,
                    ok=False,
                    error_reason="network_error",
                )

        if resp is None:
            return GeminiVisionReply(
                text="Gemini Vision timed out. Please try asking again.",
                model=used_model,
                ok=False,
                error_reason=last_exc or "timeout",
            )

        if resp.status_code in (400, 401, 403):
            return GeminiVisionReply(
                text="Your Gemini API key was rejected. Please check the key in Settings.",
                model=self._model,
                ok=False,
                error_reason="auth_error",
            )
        if resp.status_code == 429:
            return GeminiVisionReply(
                text="Gemini API rate limit or quota was reached. Please wait a moment and try again.",
                model=self._model,
                ok=False,
                error_reason="rate_limited",
            )
        if resp.status_code == 503:
            return GeminiVisionReply(
                text="Gemini is temporarily busy due to high demand. Please try again in a moment.",
                model=self._model,
                ok=False,
                error_reason="service_unavailable",
            )
        if resp.status_code >= 400:
            return GeminiVisionReply(
                text=f"Gemini Vision returned an error (HTTP {resp.status_code}).",
                model=self._model,
                ok=False,
                error_reason=f"http_{resp.status_code}",
            )

        try:
            data = resp.json()
            candidates = data.get("candidates") or []
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            text = " ".join(str(p.get("text", "")).strip() for p in parts if p.get("text")).strip()
        except Exception:
            text = ""

        if not text:
            return GeminiVisionReply(
                text="Gemini Vision did not return an answer. Please try again.",
                model=self._model,
                ok=False,
                error_reason="empty_response",
            )

        return GeminiVisionReply(text=text, model=self._model, ok=True)

from __future__ import annotations

import base64
import io
import secrets
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import qrcode


class PairingState(StrEnum):
    PENDING = "pending"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass(slots=True)
class PairingSession:
    token: str
    short_code: str
    created_at: datetime
    expires_at: datetime
    state: PairingState = PairingState.PENDING
    consent_granted: bool = False
    redeemed: bool = False
    camera_source_id: str | None = None


class PhonePairingManager:
    """
    Manages short-lived, single-use pairing tokens for remote phone cameras.

    Enforces:
    - Token expiry
    - Single-use redemption (reused tokens after disconnect are rejected)
    - Explicit camera consent before stream activation
    """

    def __init__(self, ttl_seconds: int = 180) -> None:
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, PairingSession] = {}
        self._lock = threading.Lock()

    def create_pairing(
        self, base_url: str, now: datetime | None = None
    ) -> tuple[PairingSession, str, str]:
        ref = now or datetime.now(UTC)
        token = secrets.token_urlsafe(16)
        short_code = f"{secrets.randbelow(900000) + 100000}"
        sess = PairingSession(
            token=token,
            short_code=short_code,
            created_at=ref,
            expires_at=ref + timedelta(seconds=self.ttl_seconds),
        )
        with self._lock:
            self._sessions[token] = sess

        join_url = f"{base_url.rstrip('/')}/phone/{token}"
        qr_img = qrcode.make(join_url)
        buf = io.BytesIO()
        qr_img.save(buf, format="PNG")
        qr_data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        return sess, join_url, qr_data_uri

    def get_session(self, token: str, now: datetime | None = None) -> PairingSession | None:
        ref = now or datetime.now(UTC)
        with self._lock:
            sess = self._sessions.get(token)
            if sess is None:
                return None
            if sess.state == PairingState.PENDING and ref > sess.expires_at:
                sess.state = PairingState.EXPIRED
            return sess

    def connect(
        self,
        token: str,
        consent_granted: bool,
        now: datetime | None = None,
    ) -> tuple[bool, str]:
        ref = now or datetime.now(UTC)
        with self._lock:
            sess = self._sessions.get(token)
            if sess is None:
                return False, "Invalid pairing token."
            if ref > sess.expires_at and not sess.redeemed:
                sess.state = PairingState.EXPIRED
                return False, "Pairing token has expired. Please generate a new QR code."
            if sess.redeemed or sess.state in (PairingState.DISCONNECTED, PairingState.REJECTED):
                return (
                    False,
                    "This pairing token has already been used or disconnected. Generate a new one.",
                )
            if not consent_granted:
                sess.state = PairingState.REJECTED
                sess.redeemed = True
                return False, "Camera permission was not granted on the phone."

            sess.redeemed = True
            sess.consent_granted = True
            sess.state = PairingState.CONNECTED
            sess.camera_source_id = f"phone-{sess.short_code}"
            return True, "Phone camera connected."

    def is_stream_authorized(self, token: str) -> bool:
        with self._lock:
            sess = self._sessions.get(token)
            return bool(sess and sess.state == PairingState.CONNECTED and sess.consent_granted)

    def disconnect(self, token: str) -> bool:
        with self._lock:
            sess = self._sessions.get(token)
            if sess is None:
                return False
            sess.state = PairingState.DISCONNECTED
            sess.redeemed = True
            return True

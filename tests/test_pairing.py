from __future__ import annotations

from datetime import UTC, datetime, timedelta

from services.core.pairing import PairingState, PhonePairingManager


def test_pairing_token_expiry_rejected() -> None:
    mgr = PhonePairingManager(ttl_seconds=60)
    t0 = datetime(2026, 11, 15, 9, 0, 0, tzinfo=UTC)
    sess, _, qr = mgr.create_pairing("http://localhost:8765", now=t0)
    assert qr.startswith("data:image/png;base64,")

    ok, msg = mgr.connect(sess.token, consent_granted=True, now=t0 + timedelta(seconds=61))
    assert ok is False
    assert "expired" in msg.lower()
    assert mgr.get_session(sess.token, now=t0 + timedelta(seconds=61)).state == PairingState.EXPIRED  # type: ignore[union-attr]


def test_pairing_token_cannot_be_reused_after_disconnect() -> None:
    mgr = PhonePairingManager(ttl_seconds=180)
    sess, _, _ = mgr.create_pairing("http://localhost:8765")

    ok, _ = mgr.connect(sess.token, consent_granted=True)
    assert ok is True
    assert mgr.is_stream_authorized(sess.token) is True

    assert mgr.disconnect(sess.token) is True
    assert mgr.is_stream_authorized(sess.token) is False

    # Reuse must be rejected
    ok_reuse, msg = mgr.connect(sess.token, consent_granted=True)
    assert ok_reuse is False
    assert "already been used" in msg.lower()


def test_pairing_permission_denied_locks_session() -> None:
    mgr = PhonePairingManager(ttl_seconds=180)
    sess, _, _ = mgr.create_pairing("http://localhost:8765")

    ok, msg = mgr.connect(sess.token, consent_granted=False)
    assert ok is False
    assert "permission" in msg.lower()
    assert mgr.is_stream_authorized(sess.token) is False

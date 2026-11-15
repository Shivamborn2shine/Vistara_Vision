from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from packages.vision.contracts import BoundingBox, Detection
from packages.vision.spatial import narrate_scene
from services.core.config import Settings
from services.core.main import create_app


def test_spatial_narration_describes_held_object_relation() -> None:
    person = Detection("person", BoundingBox(0.2, 0.2, 0.8, 0.95), 0.85, "v2")
    note = Detection("Rs 100 banknote", BoundingBox(0.35, 0.45, 0.65, 0.70), 0.92, "v2")
    text = narrate_scene([person, note])
    assert "person" in text.lower()
    assert "rs 100 banknote" in text.lower()
    assert "held or in front of the person" in text.lower()


def test_conversational_presence_color_and_help_commands(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path))
    client = TestClient(app)

    client.post("/api/v1/cameras/start", json={"source_id": "mock-cam-0"})

    r_help = client.post("/api/v1/commands", json={"text": "Help"})
    assert r_help.status_code == 200
    assert "what is in front of me" in r_help.json()["spoken_response"].lower()

    r_color = client.post("/api/v1/commands", json={"text": "What color is this?"})
    assert r_color.status_code == 200
    assert "color" in r_color.json()

    r_pres = client.post("/api/v1/commands", json={"text": "Is there a laptop?"})
    assert r_pres.status_code == 200
    assert r_pres.json()["present"] is False

    r_known = client.post("/api/v1/commands", json={"text": "What objects do you know?"})
    assert r_known.status_code == 200
    assert "learned objects" in r_known.json()["spoken_response"].lower() or "not taught" in r_known.json()["spoken_response"].lower()


def test_general_voice_commands(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path))
    client = TestClient(app)

    r_hi = client.post("/api/v1/commands", json={"text": "Hello"})
    assert r_hi.status_code == 200
    assert r_hi.json()["intent"] == "greeting"

    r_time = client.post("/api/v1/commands", json={"text": "What time is it?"})
    assert r_time.status_code == 200
    assert r_time.json()["intent"] == "get_time"
    assert "it is " in r_time.json()["spoken_response"].lower()

    client.post("/api/v1/cameras/start", json={"source_id": "mock-cam-0"})
    r_light = client.post("/api/v1/commands", json={"text": "How is the light?"})
    assert r_light.status_code == 200
    assert r_light.json()["intent"] == "check_lighting"


def test_gemini_vision_config_and_open_ended_command_routing(tmp_path: Path, monkeypatch) -> None:
    class _FakeResponse:
        status_code = 200

        def json(self) -> dict:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "There is a blue water bottle in front of you."}]
                        }
                    }
                ]
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, *args, **kwargs) -> _FakeResponse:
            return _FakeResponse()

    import packages.vision.gemini_vision as gv

    monkeypatch.setattr(gv.httpx, "Client", _FakeClient)

    app = create_app(Settings(data_dir=tmp_path))
    client = TestClient(app)

    assert client.get("/api/v1/ai/config").json()["configured"] is False
    cfg = client.post(
        "/api/v1/ai/config", json={"api_key": "test-key-123", "model": "gemini-2.5-flash"}
    )
    assert cfg.status_code == 200
    assert cfg.json()["configured"] is True

    client.post("/api/v1/cameras/start", json={"source_id": "mock-cam-0"})
    r_cmd = client.post("/api/v1/commands", json={"text": "What medicine bottle am I holding?"})
    assert r_cmd.status_code == 200
    assert r_cmd.json()["intent"] == "ai_vision"
    assert "blue water bottle" in r_cmd.json()["spoken_response"].lower()

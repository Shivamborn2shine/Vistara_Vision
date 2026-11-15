from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from services.core.config import Settings
from services.core.main import create_app


def _make_client(tmp_path: Path) -> TestClient:
    app = create_app(Settings(data_dir=tmp_path))
    return TestClient(app)


def test_api_v1_end_to_end_voice_and_vision_flow(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # Health & cameras
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    cams = client.get("/api/v1/cameras").json()
    assert any(c["source_id"] == "mock-cam-0" for c in cams)

    # Start mock camera explicitly
    r = client.post("/api/v1/cameras/start", json={"source_id": "mock-cam-0"})
    assert r.status_code == 200
    assert r.json()["status"] == "active"

    # Describe scene
    r = client.post("/api/v1/vision/describe")
    assert r.status_code == 200
    assert "spoken_response" in r.json()

    # Teach object via voice command
    r = client.post("/api/v1/commands", json={"text": "Learn this as my blue bottle"})
    assert r.status_code == 200
    assert r.json()["label"] == "blue bottle"

    # Describe again -> should recognize personal object and log observation
    r = client.post("/api/v1/commands", json={"text": "What is in front of me?"})
    assert r.status_code == 200
    assert "blue bottle" in r.json()["spoken_response"].lower()

    # Ask memory
    r = client.post("/api/v1/commands", json={"text": "Where was my blue bottle?"})
    assert r.status_code == 200
    assert r.json()["found"] is True
    assert "last observed" in r.json()["spoken_response"].lower()

    # Forget last observation
    r = client.post("/api/v1/commands", json={"text": "Forget that observation"})
    assert r.status_code == 200
    assert "forgot" in r.json()["spoken_response"].lower()

    # Stop camera
    r = client.post("/api/v1/commands", json={"text": "Stop camera"})
    assert r.status_code == 200
    assert r.json()["status"] == "stopped"

    # Diagnostics
    diag = client.get("/api/v1/diagnostics")
    assert diag.status_code == 200
    assert diag.json()["cloud_vision_enabled"] is False

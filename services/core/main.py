from __future__ import annotations

import asyncio
import socket
import threading
import time
from datetime import datetime
from typing import Annotated, Any

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from packages.memory.answering import MemoryAnswerEngine
from packages.memory.repository import MemoryRepository
from packages.speech.commands import CommandIntent, parse_voice_command
from packages.vision.camera import CameraManager, PhoneStreamCameraSource
from packages.vision.coco_labels import OBJECT_KNOWLEDGE_HINTS
from packages.vision.contracts import CameraFrame, CameraInfo, CameraStatus, Detection
from packages.vision.currency import INRCurrencyAnalyzer
from packages.vision.detector import (
    OnnxCocoObjectDetector,
    assess_frame_lighting,
    describe_relative_position,
)
from packages.vision.encoder import LocalHistogramVisualEncoder
from packages.vision.gemini_vision import GeminiVisionAssistant
from packages.vision.ocr import LocalOCREngine
from packages.vision.pipeline import BoundedFrameQueue, PerceptionPipeline
from packages.vision.spatial import describe_dominant_color, narrate_scene
from services.core.config import Settings
from services.core.pairing import PhonePairingManager

# --- Pydantic Schemas ---


class StartCameraRequest(BaseModel):
    source_id: str | None = None


class TeachObjectRequest(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    aliases: list[str] = Field(default_factory=list)
    owner_note: str | None = None


class UpdateObjectRequest(BaseModel):
    label: str | None = None
    aliases: list[str] | None = None
    owner_note: str | None = None


class CorrectObservationRequest(BaseModel):
    corrected_label: str = Field(min_length=1, max_length=120)


class VoiceCommandRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class PairingConnectRequest(BaseModel):
    consent_granted: bool


class PublicUrlRequest(BaseModel):
    public_url: str | None = None


class AiConfigRequest(BaseModel):
    api_key: str = Field(default="", max_length=256)
    model: str | None = Field(default=None, max_length=80)


class AskAiRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class AppContainer:
    """Holds modular services for the Vistara Vision local monolith."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.camera_manager = CameraManager()
        self.detector = OnnxCocoObjectDetector()
        self.encoder = LocalHistogramVisualEncoder()
        self.currency = INRCurrencyAnalyzer()
        self.ocr = LocalOCREngine()
        self.gemini = GeminiVisionAssistant(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            config_path=settings.ai_config_path,
        )
        self.pipeline = PerceptionPipeline(
            detector=self.detector,
            encoder=self.encoder,
            min_confidence=0.42,
        )
        self.frame_queue = BoundedFrameQueue(maxsize=2)
        self.memory = MemoryRepository(db_path=settings.db_path)
        self.answering = MemoryAnswerEngine(repo=self.memory)
        self.pairing = PhonePairingManager(ttl_seconds=settings.pairing_ttl_seconds)
        self.active_session_id: str | None = None
        self.last_spoken_response: str = "Vistara Vision is ready."
        self.public_url: str | None = settings.public_url
        self.start_time = time.monotonic()
        self._currency_lock = threading.Lock()
        self.currency_seq: int = 0

    def next_currency_seq(self) -> int:
        with self._currency_lock:
            self.currency_seq += 1
            return self.currency_seq


def _get_lan_ip() -> str:
    """Return the laptop's primary LAN IPv4 address for phone pairing on the same Wi-Fi."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return str(ip)
    except OSError:
        pass
    return "127.0.0.1"


def _phone_sender_html(token: str, short_code: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Vistara Vision — Phone Camera Pairing</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 1.25rem; }}
    .card {{ max-width: 460px; margin: 0 auto; background: #1e293b; border-radius: 14px; padding: 1.25rem; border: 2px solid #38bdf8; }}
    h1 {{ font-size: 1.35rem; margin-top: 0; }}
    button {{ width: 100%; padding: 1rem; font-size: 1.1rem; font-weight: 700; border-radius: 10px; border: none; cursor: pointer; margin-top: 0.75rem; }}
    .primary {{ background: #38bdf8; color: #0f172a; }}
    .danger {{ background: #ef4444; color: #fff; }}
    video {{ width: 100%; border-radius: 10px; margin-top: 1rem; background: #000; }}
    .status {{ margin-top: 0.75rem; padding: 0.75rem; border-radius: 8px; background: #0f172a; font-size: 0.95rem; }}
  </style>
</head>
<body>
  <main class="card">
    <h1>Vistara Vision Phone Camera</h1>
    <p>Pairing code: <strong>{short_code}</strong></p>
    <p>Your camera will only turn on after you explicitly tap <strong>Allow Camera &amp; Connect</strong> below.</p>
    <button id="connectBtn" class="primary">Allow Camera &amp; Connect</button>
    <label id="fallbackLabel" class="primary" style="display:none; text-align:center; padding:1rem; border-radius:10px; font-weight:700; cursor:pointer; margin-top:0.75rem;">
      Capture &amp; Send Photo
      <input id="photoInput" type="file" accept="image/*" capture="environment" style="display:none;" />
    </label>
    <button id="stopBtn" class="danger" style="display:none;">Stop Streaming &amp; Disconnect</button>
    <div id="status" class="status" role="status" aria-live="polite">Waiting for your permission.</div>
    <video id="preview" autoplay playsinline muted style="display:none;"></video>
    <canvas id="canvas" style="display:none;"></canvas>
  </main>
  <script>
    const token = {token!r};
    const statusEl = document.getElementById('status');
    const connectBtn = document.getElementById('connectBtn');
    const fallbackLabel = document.getElementById('fallbackLabel');
    const photoInput = document.getElementById('photoInput');
    const stopBtn = document.getElementById('stopBtn');
    const video = document.getElementById('preview');
    const canvas = document.getElementById('canvas');
    let stream = null;
    let timer = null;

    async function authorizeConnection() {{
      const res = await fetch(`/api/v1/pairing/connect/${{token}}`, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ consent_granted: true }})
      }});
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || data.message || 'Connection rejected');
    }}

    connectBtn.onclick = async () => {{
      try {{
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
          await authorizeConnection();
          connectBtn.style.display = 'none';
          fallbackLabel.style.display = 'block';
          stopBtn.style.display = 'block';
          statusEl.textContent = 'Connected (HTTP photo mode). Tap Capture & Send Photo to stream frames.';
          return;
        }}
        stream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: 'environment' }}, audio: false }});
        await authorizeConnection();
        video.srcObject = stream;
        video.style.display = 'block';
        connectBtn.style.display = 'none';
        stopBtn.style.display = 'block';
        statusEl.textContent = 'Connected. Streaming frames to your laptop.';
        timer = setInterval(sendFrame, 350);
      }} catch (err) {{
        statusEl.textContent = 'Could not connect: ' + err.message;
      }}
    }};

    photoInput.onchange = async (e) => {{
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const form = new FormData();
      form.append('file', file, 'frame.jpg');
      const r = await fetch(`/api/v1/pairing/frame/${{token}}`, {{ method: 'POST', body: form }});
      statusEl.textContent = r.ok ? 'Photo sent to laptop!' : 'Failed to send photo (session ended).';
    }};

    async function sendFrame() {{
      if (!stream || video.videoWidth === 0) return;
      canvas.width = 640;
      canvas.height = Math.round((video.videoHeight / video.videoWidth) * 640) || 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(async (blob) => {{
        if (!blob) return;
        const form = new FormData();
        form.append('file', blob, 'frame.jpg');
        const r = await fetch(`/api/v1/pairing/frame/${{token}}`, {{ method: 'POST', body: form }});
        if (!r.ok) {{
          stopAll('Stream disconnected by laptop or expired.');
        }}
      }}, 'image/jpeg', 0.78);
    }}

    async function stopAll(msg) {{
      if (timer) clearInterval(timer);
      if (stream) stream.getTracks().forEach(t => t.stop());
      stream = null;
      await fetch(`/api/v1/pairing/disconnect/${{token}}`, {{ method: 'POST' }});
      video.style.display = 'none';
      stopBtn.style.display = 'none';
      connectBtn.style.display = 'none';
      statusEl.textContent = msg || 'Disconnected. You may close this tab.';
    }}

    stopBtn.onclick = () => stopAll('Stopped by user.');
  </script>
</body>
</html>"""


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()
    c = AppContainer(cfg)

    app = FastAPI(title="Vistara Vision API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_private_network_header(request: Request, call_next: Any) -> Response:
        response: Response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    app.state.container = c

    # --- Health & Diagnostics (FR14) ---

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "app": cfg.app_name,
            "api_version": cfg.api_version,
            "camera_id": c.camera_manager.active_camera_id(),
            "camera_status": c.camera_manager.active_status().value,
        }

    @app.get("/api/v1/diagnostics")
    def diagnostics() -> dict[str, Any]:
        return {
            "uptime_seconds": round(time.monotonic() - c.start_time, 1),
            "camera_id": c.camera_manager.active_camera_id(),
            "camera_status": c.camera_manager.active_status().value,
            "dropped_frames": c.frame_queue.dropped_frames,
            "cloud_vision_enabled": cfg.cloud_vision_enabled,
            "models": {
                "detector": c.detector.model_version,
                "encoder": c.encoder.encoder_version,
                "currency": c.currency.model_version,
                "ocr": c.ocr.engine_version,
            },
            "learned_objects_count": len(c.memory.list_objects()),
        }

    # --- Camera Controls (FR01) ---

    @app.get("/api/v1/cameras")
    def list_cameras() -> list[dict[str, Any]]:
        return [
            {
                "source_id": cam.source_id,
                "name": cam.name,
                "source_type": cam.source_type,
                "available": cam.available,
            }
            for cam in c.camera_manager.list_cameras()
        ]

    @app.post("/api/v1/cameras/start")
    def start_camera(req: StartCameraRequest) -> dict[str, Any]:
        cam_id, status = c.camera_manager.start_camera(req.source_id)
        if status != CameraStatus.ACTIVE:
            raise HTTPException(
                status_code=503,
                detail=f"Camera {cam_id} could not start (status: {status.value}). Check camera permissions.",
            )
        c.active_session_id = c.memory.start_session(source_type=cam_id)
        msg = f"Camera started: {cam_id}."
        c.last_spoken_response = msg
        return {"camera_id": cam_id, "status": status.value, "spoken_response": msg}

    @app.post("/api/v1/cameras/stop")
    def stop_camera() -> dict[str, Any]:
        stopped_id, status = c.camera_manager.stop_camera()
        c.frame_queue.clear()
        if c.active_session_id:
            c.memory.end_session(c.active_session_id)
            c.active_session_id = None
        msg = "Camera stopped." if stopped_id else "Camera was already stopped."
        c.last_spoken_response = msg
        return {"camera_id": stopped_id, "status": status.value, "spoken_response": msg}

    @app.get("/api/v1/cameras/preview")
    def camera_preview() -> Response:
        frame = c.camera_manager.capture_frame()
        if frame is None:
            raise HTTPException(status_code=404, detail="No active camera frame.")
        return Response(content=frame.jpeg_bytes, media_type="image/jpeg")

    @app.get("/api/v1/cameras/stream")
    async def camera_stream() -> StreamingResponse:
        async def _mjpeg_generator() -> Any:
            while c.camera_manager.active_status() == CameraStatus.ACTIVE:
                frame = c.camera_manager.capture_frame()
                if frame is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame.jpeg_bytes + b"\r\n"
                    )
                await asyncio.sleep(0.066)

        return StreamingResponse(
            _mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/api/v1/cameras/browser/frame")
    async def browser_push_frame(file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
        src = c.camera_manager.get_source("browser-cam-0")
        if not isinstance(src, PhoneStreamCameraSource):
            raise HTTPException(status_code=404, detail="Browser camera source not registered.")
        if src.status() != CameraStatus.ACTIVE:
            c.camera_manager.start_camera("browser-cam-0")
        data = await file.read()
        src.push_frame(data)
        return {"received": True, "bytes": len(data)}

    # --- Vision Actions (FR04, FR05, FR06, FR07) ---

    def _require_frame() -> Any:
        frame = c.camera_manager.capture_frame()
        if frame is None:
            raise HTTPException(
                status_code=409,
                detail="Camera is not active. Start the camera first.",
            )
        c.frame_queue.push(frame)
        return frame

    @app.post("/api/v1/vision/detect")
    def detect_live() -> dict[str, Any]:
        """Lightweight detection-only endpoint for live preview bounding boxes."""
        frame = _require_frame()
        profiles = c.memory.load_enrolled_profiles()
        detections, _ = c.pipeline.process_frame(frame, profiles=profiles, require_stability=1)
        return {
            "detections": [
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "is_personal_match": d.is_personal_match,
                    "object_id": d.object_id,
                    "position": describe_relative_position(d.box),
                    "box": {"x1": d.box.x1, "y1": d.box.y1, "x2": d.box.x2, "y2": d.box.y2},
                }
                for d in detections
            ]
        }

    @app.post("/api/v1/vision/describe")
    def describe_scene() -> dict[str, Any]:
        frame = _require_frame()
        _, lighting_issue = assess_frame_lighting(frame)
        profiles = c.memory.load_enrolled_profiles()
        detections, stable = c.pipeline.process_frame(frame, profiles=profiles, require_stability=1)

        # Smart fusion: if a handheld rectangular object ('book'/'remote') is actually an INR note,
        # upgrade its label so "What do you see?" identifies currency naturally.
        note_boxes = [
            d.box for d in detections if d.label in ("book", "remote") and not d.is_personal_match
        ]
        if note_boxes and lighting_issue is None:
            cur_res = c.currency.analyze([frame], candidate_boxes=note_boxes)
            if not cur_res.abstained and cur_res.notes:
                top_note = cur_res.notes[0]
                fused: list[Detection] = []
                replaced = False
                for d in detections:
                    if not replaced and d.label in ("book", "remote") and not d.is_personal_match:
                        fused.append(
                            Detection(
                                label=f"Rs {top_note.denomination_inr} banknote",
                                box=d.box,
                                confidence=max(d.confidence, top_note.confidence),
                                model_version=d.model_version,
                                track_id=d.track_id,
                            )
                        )
                        replaced = True
                    else:
                        fused.append(d)
                detections = fused

        for obs in stable:
            source = "user_confirmed" if obs.object_id else "model_inferred"
            c.memory.write_observation(obs, label_source=source)

        spoken = narrate_scene(detections, lighting_warning=lighting_issue)
        if c.gemini.configured:
            known_names = [o.label for o in c.memory.list_objects()]
            reply = c.gemini.ask(
                "Describe clearly what is in front of me right now.",
                frame=frame,
                known_objects=known_names,
            )
            if reply.ok:
                spoken = reply.text

        c.last_spoken_response = spoken
        return {
            "spoken_response": spoken,
            "lighting_warning": lighting_issue,
            "detections": [
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "is_personal_match": d.is_personal_match,
                    "object_id": d.object_id,
                    "position": describe_relative_position(d.box),
                    "box": {"x1": d.box.x1, "y1": d.box.y1, "x2": d.box.x2, "y2": d.box.y2},
                }
                for d in detections
            ],
        }

    @app.get("/api/v1/ai/config")
    def get_ai_config() -> dict[str, Any]:
        return {"configured": c.gemini.configured, "model": c.gemini.model}

    @app.post("/api/v1/ai/config")
    def set_ai_config(req: AiConfigRequest) -> dict[str, Any]:
        c.gemini.configure(req.api_key, model=req.model)
        return {"configured": c.gemini.configured, "model": c.gemini.model}

    @app.post("/api/v1/vision/ask")
    def ask_ai_vision(req: AskAiRequest) -> dict[str, Any]:
        frame: CameraFrame | None = None
        if c.camera_manager.active_status() == CameraStatus.ACTIVE:
            frame = c.camera_manager.capture_frame()
        known_names = [o.label for o in c.memory.list_objects()]
        reply = c.gemini.ask(req.question, frame=frame, known_objects=known_names)
        c.last_spoken_response = reply.text
        return {
            "spoken_response": reply.text,
            "model": reply.model,
            "ok": reply.ok,
            "error_reason": reply.error_reason,
        }

    @app.post("/api/v1/vision/ocr")
    def read_text() -> dict[str, Any]:
        frame = _require_frame()
        res = c.ocr.read_text(frame)
        spoken = res.guidance if res.abstained and res.guidance else f"Text reads: {res.full_text}"
        c.last_spoken_response = spoken
        return {
            "spoken_response": spoken,
            "full_text": res.full_text,
            "confidence": res.confidence,
            "abstained": res.abstained,
            "engine_version": res.engine_version,
        }

    @app.post("/api/v1/vision/currency")
    def analyze_currency() -> dict[str, Any]:
        my_seq = c.next_currency_seq()
        frame = _require_frame()
        # Use detector boxes (e.g. 'book' / rectangular held objects) as priority note crops
        detections = c.detector.detect(frame)
        candidate_boxes = [
            d.box for d in detections if d.label in ("book", "cell phone", "remote", "handbag")
        ]
        res = c.currency.analyze(
            [frame],
            candidate_boxes=candidate_boxes or None,
            cancel_check=lambda: c.currency_seq != my_seq,
        )
        spoken = f"{res.guidance} {res.authenticity_disclaimer}"
        c.last_spoken_response = spoken
        return {
            "spoken_response": spoken,
            "distinct_count": res.distinct_count,
            "total_inr": res.total_inr,
            "abstained": res.abstained,
            "uncertainty_reason": res.uncertainty_reason,
            "authenticity_disclaimer": res.authenticity_disclaimer,
            "notes": [
                {"denomination_inr": n.denomination_inr, "confidence": n.confidence}
                for n in res.notes
            ],
        }

    # --- Teachable Objects (FR08, FR09) ---

    @app.get("/api/v1/objects")
    def list_objects() -> list[dict[str, Any]]:
        return [
            {
                "object_id": o.object_id,
                "label": o.label,
                "aliases": o.aliases,
                "owner_note": o.owner_note,
                "embedding_count": o.embedding_count,
                "created_at": o.created_at.isoformat(),
                "edited_at": o.edited_at.isoformat(),
            }
            for o in c.memory.list_objects()
        ]

    @app.post("/api/v1/objects/teach")
    def teach_object(req: TeachObjectRequest) -> dict[str, Any]:
        frame = _require_frame()
        emb = c.encoder.encode(frame)
        rec = c.memory.enroll_object(
            label=req.label,
            embeddings=[(emb.values, emb.encoder_version)],
            aliases=req.aliases,
            owner_note=req.owner_note,
        )
        spoken = f"Learned {rec.label}. I will recognize it in future scenes."
        c.last_spoken_response = spoken
        return {
            "object_id": rec.object_id,
            "label": rec.label,
            "embedding_count": rec.embedding_count,
            "spoken_response": spoken,
        }

    @app.patch("/api/v1/objects/{object_id}")
    def update_object(object_id: str, req: UpdateObjectRequest) -> dict[str, Any]:
        rec = c.memory.update_object(
            object_id=object_id,
            label=req.label,
            aliases=req.aliases,
            owner_note=req.owner_note,
        )
        if rec is None:
            raise HTTPException(status_code=404, detail="Object not found.")
        return {"object_id": rec.object_id, "label": rec.label, "aliases": rec.aliases}

    @app.delete("/api/v1/objects/{object_id}")
    def delete_object(object_id: str) -> dict[str, Any]:
        ok = c.memory.delete_object(object_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Object not found.")
        return {"deleted": True, "object_id": object_id}

    # --- Episodic Memory (FR10, FR11, FR13) ---

    @app.get("/api/v1/memory/observations")
    def list_observations(limit: int = 20) -> list[dict[str, Any]]:
        return [
            {
                "event_id": r.event_id,
                "object_id": r.object_id,
                "label": r.label,
                "label_source": r.label_source,
                "timestamp_utc": r.timestamp_utc.isoformat(),
                "camera_id": r.camera_id,
                "room_id": r.room_id,
                "landmark_relation": r.landmark_relation,
                "confidence": r.confidence,
            }
            for r in c.memory.list_recent_observations(limit=limit)
        ]

    @app.get("/api/v1/memory/last-seen")
    def query_last_seen(item: str) -> dict[str, Any]:
        ans = c.answering.answer_last_seen(item)
        c.last_spoken_response = ans.spoken_answer
        return {
            "found": ans.found,
            "spoken_response": ans.spoken_answer,
            "conflicting_sightings": ans.conflicting_sightings,
            "latest_observation": (
                {
                    "event_id": ans.latest_observation.event_id,
                    "label": ans.latest_observation.label,
                    "timestamp_utc": ans.latest_observation.timestamp_utc.isoformat(),
                    "landmark_relation": ans.latest_observation.landmark_relation,
                    "camera_id": ans.latest_observation.camera_id,
                    "confidence": ans.latest_observation.confidence,
                }
                if ans.latest_observation
                else None
            ),
        }

    @app.post("/api/v1/memory/observations/{event_id}/correct")
    def correct_observation(event_id: str, req: CorrectObservationRequest) -> dict[str, Any]:
        rec = c.memory.correct_observation(event_id, req.corrected_label)
        if rec is None:
            raise HTTPException(status_code=404, detail="Observation not found.")
        return {"event_id": rec.event_id, "label": rec.label, "label_source": rec.label_source}

    @app.delete("/api/v1/memory/observations/{event_id}")
    def delete_observation(event_id: str) -> dict[str, Any]:
        if not c.memory.delete_observation(event_id):
            raise HTTPException(status_code=404, detail="Observation not found.")
        return {"deleted": True, "event_id": event_id}

    @app.delete("/api/v1/memory")
    def clear_all_memory() -> dict[str, Any]:
        c.memory.clear_all_memory()
        return {
            "cleared": True,
            "spoken_response": "All saved objects and observations have been cleared.",
        }

    @app.get("/api/v1/memory/export")
    def export_memory() -> dict[str, Any]:
        return c.memory.export_memory()

    # --- Unified Voice/Text Command Router (FR12) ---

    @app.post("/api/v1/commands")
    def handle_command(req: VoiceCommandRequest, request: Request) -> dict[str, Any]:
        parsed = parse_voice_command(req.text)
        intent = parsed.intent

        if intent == CommandIntent.REPEAT:
            return {"intent": intent.value, "spoken_response": c.last_spoken_response}

        if intent == CommandIntent.STOP_SPEAKING:
            return {"intent": intent.value, "spoken_response": "Speech stopped."}

        if intent == CommandIntent.START_VISION:
            return {"intent": intent.value, **start_camera(StartCameraRequest())}

        if intent == CommandIntent.STOP_CAMERA:
            return {"intent": intent.value, **stop_camera()}

        if intent == CommandIntent.DESCRIBE_SCENE:
            return {"intent": intent.value, **describe_scene()}

        if intent == CommandIntent.READ_TEXT:
            return {"intent": intent.value, **read_text()}

        if intent == CommandIntent.COUNT_CURRENCY:
            return {"intent": intent.value, **analyze_currency()}

        if intent == CommandIntent.TEACH_OBJECT:
            if not parsed.argument:
                msg = "What should I call this object? Say, learn this as my bottle."
                return {"intent": intent.value, "needs_argument": True, "spoken_response": msg}
            return {
                "intent": intent.value,
                **teach_object(TeachObjectRequest(label=parsed.argument)),
            }

        if intent == CommandIntent.CORRECT_OBJECT:
            recent = c.memory.list_recent_observations(limit=1)
            if not recent or not parsed.argument:
                msg = "I do not have a recent observation to correct."
                return {"intent": intent.value, "spoken_response": msg}
            c.memory.correct_observation(recent[0].event_id, parsed.argument)
            msg = f"Updated the last observation to {parsed.argument}."
            c.last_spoken_response = msg
            return {"intent": intent.value, "spoken_response": msg}

        if intent == CommandIntent.ASK_MEMORY:
            if not parsed.argument:
                return {
                    "intent": intent.value,
                    "spoken_response": "Which item are you looking for?",
                }
            return {"intent": intent.value, **query_last_seen(parsed.argument)}

        if intent == CommandIntent.FORGET_LAST:
            recent = c.memory.list_recent_observations(limit=1)
            if not recent:
                return {
                    "intent": intent.value,
                    "spoken_response": "There are no recent observations to forget.",
                }
            c.memory.delete_observation(recent[0].event_id)
            msg = f"Forgot the last observation of {recent[0].label}."
            c.last_spoken_response = msg
            return {"intent": intent.value, "spoken_response": msg}

        if intent == CommandIntent.PAIR_PHONE:
            return {"intent": intent.value, **create_pairing(request)}

        if intent == CommandIntent.DESCRIBE_COLOR:
            frame = _require_frame()
            color_name = describe_dominant_color(frame)
            msg = f"The main color in front of the camera looks {color_name}."
            c.last_spoken_response = msg
            return {"intent": intent.value, "color": color_name, "spoken_response": msg}

        if intent == CommandIntent.CHECK_PRESENCE:
            target = (parsed.argument or "").strip()
            if not target:
                return {"intent": intent.value, "spoken_response": "What item should I look for?"}
            frame = _require_frame()
            profiles = c.memory.load_enrolled_profiles()
            detections, _ = c.pipeline.process_frame(frame, profiles=profiles, require_stability=1)
            matches = [d for d in detections if target in d.label.lower()]
            if matches:
                first = matches[0]
                msg = (
                    f"Yes, I see {len(matches)} {first.label} "
                    f"({describe_relative_position(first.box)}, {int(first.confidence * 100)}% confidence)."
                )
                c.last_spoken_response = msg
                return {"intent": intent.value, "present": True, "spoken_response": msg}
            mem_ans = c.answering.answer_last_seen(target)
            msg = (
                f"I do not see {target} in front of the camera right now. "
                + (mem_ans.spoken_answer if mem_ans.found else "It is not in saved memory either.")
            )
            c.last_spoken_response = msg
            return {"intent": intent.value, "present": False, "spoken_response": msg}

        if intent == CommandIntent.LIST_KNOWN:
            objs = c.memory.list_objects()
            recent = c.memory.list_recent_observations(limit=5)
            obj_part = (
                "Learned objects: " + ", ".join(o.label for o in objs) + "."
                if objs
                else "You have not taught any personal objects yet."
            )
            obs_part = (
                " Recent sightings: " + ", ".join(r.label for r in recent[:3]) + "."
                if recent
                else ""
            )
            msg = obj_part + obs_part
            c.last_spoken_response = msg
            return {"intent": intent.value, "spoken_response": msg}

        if intent == CommandIntent.GREETING:
            msg = (
                "Hello! Vistara Vision is ready. Hold the Speak button and ask: "
                "What is in front of me, Identify currency, Read this, or What time is it."
            )
            c.last_spoken_response = msg
            return {"intent": intent.value, "spoken_response": msg}

        if intent == CommandIntent.GET_TIME:
            now = datetime.now()
            msg = now.strftime("It is %I:%M %p on %A, %B %d, %Y.").replace(" 0", " ")
            c.last_spoken_response = msg
            return {"intent": intent.value, "spoken_response": msg}

        if intent == CommandIntent.CHECK_LIGHTING:
            frame = _require_frame()
            brightness, issue = assess_frame_lighting(frame)
            msg = issue or f"Lighting looks good with brightness level {int(brightness)} out of 255."
            c.last_spoken_response = msg
            return {"intent": intent.value, "spoken_response": msg}

        if intent == CommandIntent.HELP:
            msg = (
                "You can say: What is in front of me, Read this, Identify currency, "
                "What color is this, How is the light, What time is it, Is there a bottle, "
                "Learn this as my bottle, Where was my keys, Repeat, Stop speaking, "
                "Start vision, Connect phone, or Stop camera."
            )
            return {"intent": intent.value, "spoken_response": msg}

        if c.gemini.configured:
            ai_frame: CameraFrame | None = None
            if c.camera_manager.active_status() == CameraStatus.ACTIVE:
                ai_frame = c.camera_manager.capture_frame()
            known_names = [o.label for o in c.memory.list_objects()]
            reply = c.gemini.ask(req.text, frame=ai_frame, known_objects=known_names)
            c.last_spoken_response = reply.text
            return {
                "intent": "ai_vision",
                "spoken_response": reply.text,
                "model": reply.model,
                "ok": reply.ok,
            }

        msg = (
            "I did not understand that command. Say Help to hear everything I can do, or add "
            "a Gemini API key in Settings to ask any general or visual AI question."
        )
        return {"intent": CommandIntent.UNKNOWN.value, "spoken_response": msg}

    # --- Phone Pairing & Remote Camera Stream (FR02, FR03) ---

    @app.post("/api/v1/pairing/public-url")
    def set_public_url(req: PublicUrlRequest, request: Request) -> dict[str, Any]:
        # Only allow local launcher on loopback to set the tunnel public URL
        client_host = request.client.host if request.client else ""
        if client_host not in ("127.0.0.1", "::1", "testclient"):
            raise HTTPException(status_code=403, detail="Only local launcher may set public_url.")
        c.public_url = req.public_url.rstrip("/") if req.public_url else None
        return {"public_url": c.public_url}

    @app.post("/api/v1/pairing/create")
    def create_pairing(request: Request) -> dict[str, Any]:
        if c.public_url:
            base_url = c.public_url.rstrip("/")
        else:
            base_url = str(request.base_url).rstrip("/")
            if "127.0.0.1" in base_url or "localhost" in base_url:
                lan_ip = _get_lan_ip()
                base_url = base_url.replace("127.0.0.1", lan_ip).replace("localhost", lan_ip)
        sess, join_url, qr_data_uri = c.pairing.create_pairing(base_url=base_url)
        msg = f"Phone pairing ready. Scan the QR code or open the link with code {sess.short_code}."
        c.last_spoken_response = msg
        return {
            "token": sess.token,
            "short_code": sess.short_code,
            "join_url": join_url,
            "qr_data_uri": qr_data_uri,
            "expires_at": sess.expires_at.isoformat(),
            "state": sess.state.value,
            "spoken_response": msg,
        }

    @app.get("/api/v1/pairing/status/{token}")
    def pairing_status(token: str) -> dict[str, Any]:
        sess = c.pairing.get_session(token)
        if sess is None:
            raise HTTPException(status_code=404, detail="Pairing session not found.")
        return {
            "token": sess.token,
            "short_code": sess.short_code,
            "state": sess.state.value,
            "consent_granted": sess.consent_granted,
            "camera_source_id": sess.camera_source_id,
            "expires_at": sess.expires_at.isoformat(),
        }

    @app.post("/api/v1/pairing/connect/{token}")
    def pairing_connect(token: str, req: PairingConnectRequest) -> dict[str, Any]:
        ok, msg = c.pairing.connect(token=token, consent_granted=req.consent_granted)
        if not ok:
            raise HTTPException(status_code=403, detail=msg)
        sess = c.pairing.get_session(token)
        assert sess is not None and sess.camera_source_id is not None
        phone_src = PhoneStreamCameraSource(
            source_id=sess.camera_source_id, name="Paired Phone Camera"
        )
        c.camera_manager.register_source(
            phone_src,
            CameraInfo(
                source_id=sess.camera_source_id,
                name=f"Paired Phone ({sess.short_code})",
                source_type="phone_webrtc",
                available=True,
            ),
        )
        c.camera_manager.start_camera(sess.camera_source_id)
        return {"connected": True, "camera_source_id": sess.camera_source_id, "message": msg}

    @app.post("/api/v1/pairing/frame/{token}")
    async def pairing_push_frame(
        token: str,
        file: Annotated[UploadFile, File(...)],
    ) -> dict[str, Any]:
        if not c.pairing.is_stream_authorized(token):
            raise HTTPException(
                status_code=403, detail="Phone stream is not authorized or has expired."
            )
        sess = c.pairing.get_session(token)
        assert sess is not None and sess.camera_source_id is not None
        src = c.camera_manager.get_source(sess.camera_source_id)
        if not isinstance(src, PhoneStreamCameraSource):
            raise HTTPException(status_code=404, detail="Phone camera source not active.")
        data = await file.read()
        src.push_frame(data)
        return {"received": True, "bytes": len(data)}

    @app.post("/api/v1/pairing/disconnect/{token}")
    def pairing_disconnect(token: str) -> dict[str, Any]:
        sess = c.pairing.get_session(token)
        if sess is None:
            raise HTTPException(status_code=404, detail="Pairing session not found.")
        if sess.camera_source_id:
            c.camera_manager.unregister_source(sess.camera_source_id)
        c.pairing.disconnect(token)
        return {"disconnected": True}

    @app.get("/phone/{token}", response_class=HTMLResponse)
    def phone_sender_page(token: str) -> HTMLResponse:
        sess = c.pairing.get_session(token)
        if sess is None:
            return HTMLResponse("<h1>Invalid or expired pairing link.</h1>", status_code=404)
        return HTMLResponse(_phone_sender_html(token=sess.token, short_code=sess.short_code))

    @app.websocket("/ws/status")
    async def ws_status(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.receive_text()
                await websocket.send_json(
                    {
                        "camera_id": c.camera_manager.active_camera_id(),
                        "camera_status": c.camera_manager.active_status().value,
                    }
                )
        except WebSocketDisconnect:
            pass

    return app


app = create_app()

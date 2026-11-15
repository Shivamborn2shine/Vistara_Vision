from __future__ import annotations

import io
import threading
from typing import Any

from PIL import Image, ImageDraw

from packages.vision.contracts import CameraFrame, CameraInfo, CameraSource, CameraStatus, utc_now


def _render_synthetic_jpeg(
    width: int = 640,
    height: int = 480,
    label: str = "Vistara Vision Demo",
    bg_color: tuple[int, int, int] = (24, 34, 56),
) -> bytes:
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    # Draw a simple tabletop scene with a bottle-like and book-like shape
    draw.rectangle([40, 320, 600, 440], fill=(102, 74, 52))  # desk/table
    draw.rectangle([140, 180, 210, 340], fill=(45, 125, 210))  # blue bottle
    draw.rectangle([380, 270, 500, 340], fill=(220, 220, 210))  # notebook
    draw.text((20, 20), label, fill=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


class MockCameraSource:
    """Deterministic camera source for offline demos and automated tests."""

    def __init__(self, source_id: str = "mock-cam-0", name: str = "Demo Scene Camera") -> None:
        self.source_id = source_id
        self.name = name
        self._status = CameraStatus.IDLE
        self._seq = 0
        self._lock = threading.Lock()
        self._custom_jpeg: bytes | None = None

    def set_frame_bytes(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._custom_jpeg = jpeg_bytes

    def start(self) -> CameraStatus:
        with self._lock:
            self._status = CameraStatus.ACTIVE
            return self._status

    def get_frame(self) -> CameraFrame | None:
        with self._lock:
            if self._status != CameraStatus.ACTIVE:
                return None
            self._seq += 1
            data = self._custom_jpeg or _render_synthetic_jpeg(label=f"{self.name} #{self._seq}")
            return CameraFrame(
                source_id=self.source_id,
                timestamp_utc=utc_now(),
                width=640,
                height=480,
                jpeg_bytes=data,
                sequence_id=self._seq,
            )

    def status(self) -> CameraStatus:
        with self._lock:
            return self._status

    def stop(self) -> CameraStatus:
        with self._lock:
            self._status = CameraStatus.STOPPED
            return self._status


class WebcamCameraSource:
    """OpenCV-backed local webcam source with safe fallback when OpenCV is unavailable."""

    def __init__(self, device_index: int = 0, name: str | None = None) -> None:
        self.device_index = device_index
        self.source_id = f"webcam-{device_index}"
        self.name = name or f"Laptop Webcam ({device_index})"
        self._status = CameraStatus.IDLE
        self._cap: Any = None
        self._seq = 0
        self._lock = threading.Lock()

    def start(self) -> CameraStatus:
        with self._lock:
            try:
                import cv2  # type: ignore[import-not-found]
            except ImportError:
                self._status = CameraStatus.ERROR
                return self._status

            cap = cv2.VideoCapture(self.device_index)
            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                self._status = CameraStatus.PERMISSION_DENIED
                return self._status

            self._cap = cap
            self._status = CameraStatus.ACTIVE
            return self._status

    def get_frame(self) -> CameraFrame | None:
        with self._lock:
            if self._status != CameraStatus.ACTIVE or self._cap is None:
                return None
            try:
                import cv2  # type: ignore[import-not-found]
            except ImportError:
                self._status = CameraStatus.ERROR
                return None

            ok, frame = self._cap.read()
            if not ok or frame is None:
                return None
            height, width = frame.shape[:2]
            encoded_ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not encoded_ok:
                return None
            self._seq += 1
            return CameraFrame(
                source_id=self.source_id,
                timestamp_utc=utc_now(),
                width=int(width),
                height=int(height),
                jpeg_bytes=bytes(buf),
                sequence_id=self._seq,
            )

    def status(self) -> CameraStatus:
        with self._lock:
            return self._status

    def stop(self) -> CameraStatus:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._status = CameraStatus.STOPPED
            return self._status


class PhoneStreamCameraSource:
    """Camera source fed by an authenticated paired phone browser stream."""

    def __init__(self, source_id: str = "phone-stream", name: str = "Paired Phone Camera") -> None:
        self.source_id = source_id
        self.name = name
        self._status = CameraStatus.IDLE
        self._latest_frame: CameraFrame | None = None
        self._seq = 0
        self._lock = threading.Lock()

    def start(self) -> CameraStatus:
        with self._lock:
            self._status = CameraStatus.ACTIVE
            return self._status

    def push_frame(
        self, jpeg_bytes: bytes, width: int = 640, height: int = 480
    ) -> CameraFrame | None:
        with self._lock:
            if self._status != CameraStatus.ACTIVE:
                return None
            self._seq += 1
            frame = CameraFrame(
                source_id=self.source_id,
                timestamp_utc=utc_now(),
                width=width,
                height=height,
                jpeg_bytes=jpeg_bytes,
                sequence_id=self._seq,
            )
            self._latest_frame = frame
            return frame

    def get_frame(self) -> CameraFrame | None:
        with self._lock:
            if self._status != CameraStatus.ACTIVE:
                return None
            return self._latest_frame

    def status(self) -> CameraStatus:
        with self._lock:
            return self._status

    def stop(self) -> CameraStatus:
        with self._lock:
            self._status = CameraStatus.STOPPED
            self._latest_frame = None
            return self._status


class CameraManager:
    """Coordinates local and remote CameraSource instances with accessible status reporting."""

    def __init__(self) -> None:
        self._sources: dict[str, CameraSource] = {}
        self._infos: dict[str, CameraInfo] = {}
        self._active_id: str | None = None
        self._lock = threading.Lock()

        # Always register a deterministic demo camera and laptop webcam 0
        self.register_source(
            MockCameraSource(),
            CameraInfo(
                source_id="mock-cam-0",
                name="Demo Scene Camera (Built-in)",
                source_type="mock",
                available=True,
            ),
        )
        self.register_source(
            WebcamCameraSource(device_index=0),
            CameraInfo(
                source_id="webcam-0",
                name="Laptop Webcam (0)",
                source_type="laptop_webcam",
                available=self._probe_opencv_available(),
            ),
        )

    @staticmethod
    def _probe_opencv_available() -> bool:
        try:
            import cv2  # noqa: F401

            return True
        except ImportError:
            return False

    def register_source(self, source: CameraSource, info: CameraInfo) -> None:
        with self._lock:
            self._sources[info.source_id] = source
            self._infos[info.source_id] = info

    def unregister_source(self, source_id: str) -> None:
        with self._lock:
            src = self._sources.pop(source_id, None)
            self._infos.pop(source_id, None)
            if src is not None:
                src.stop()
            if self._active_id == source_id:
                self._active_id = None

    def list_cameras(self) -> list[CameraInfo]:
        with self._lock:
            return list(self._infos.values())

    def start_camera(self, source_id: str | None = None) -> tuple[str, CameraStatus]:
        with self._lock:
            target_id = source_id or self._active_id or "webcam-0"
            if target_id not in self._sources:
                target_id = "mock-cam-0"
            if self._active_id and self._active_id != target_id:
                prev = self._sources.get(self._active_id)
                if prev is not None:
                    prev.stop()
            src = self._sources[target_id]
            status = src.start()
            if status != CameraStatus.ACTIVE and target_id != "mock-cam-0" and source_id is None:
                # Fallback to mock only if caller didn't explicitly demand a specific hardware cam
                target_id = "mock-cam-0"
                status = self._sources[target_id].start()
            if status == CameraStatus.ACTIVE:
                self._active_id = target_id
            return target_id, status

    def stop_camera(self) -> tuple[str | None, CameraStatus]:
        with self._lock:
            if not self._active_id:
                return None, CameraStatus.STOPPED
            src = self._sources.get(self._active_id)
            stopped_id = self._active_id
            self._active_id = None
            if src is None:
                return stopped_id, CameraStatus.STOPPED
            return stopped_id, src.stop()

    def active_camera_id(self) -> str | None:
        with self._lock:
            return self._active_id

    def active_status(self) -> CameraStatus:
        with self._lock:
            if not self._active_id:
                return CameraStatus.STOPPED
            src = self._sources.get(self._active_id)
            return src.status() if src else CameraStatus.STOPPED

    def capture_frame(self) -> CameraFrame | None:
        with self._lock:
            if not self._active_id:
                return None
            src = self._sources.get(self._active_id)
            if src is None:
                return None
            return src.get_frame()

    def get_source(self, source_id: str) -> CameraSource | None:
        with self._lock:
            return self._sources.get(source_id)

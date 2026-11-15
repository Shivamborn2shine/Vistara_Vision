# Vistara Vision

Voice-first assistive visual AI with teachable personal object recognition, Indian banknote denomination reading, OCR, remote phone camera pairing, and time-aware episodic memory for blind and visually impaired users.

## Architecture

- `services/core/`: FastAPI `/api/v1` local service, settings, single-use phone pairing & `/phone/{token}` sender page.
- `packages/vision/`: `CameraSource` adapters (webcam, paired phone, mock), `ObjectDetector`, `INRCurrencyAnalyzer`, `OCREngine`, `VisualEncoder`, and bounded-queue `PerceptionPipeline`.
- `packages/memory/`: SQLite + SQLAlchemy 2 persistence for teachable objects, versioned embeddings, timestamped observations, user feedback, and honest `MemoryAnswerEngine`.
- `packages/speech/`: `SpeechEngine` contracts and voice/text command intent parser.
- `apps/desktop/`: High-contrast, screen-reader-accessible React + TypeScript + Vite interface with Web Speech API integration.
- `tests/`: Automated `pytest` suite covering currency abstention & de-duplication, teachable objects, honest last-seen recall, pairing security, and `/api/v1` contracts.

## Quick Start (Windows)

### 1. Backend
```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -e ".[dev]"
.\.venv\Scripts\uvicorn.exe services.core.main:app --host 127.0.0.1 --port 8765 --reload
```
Optional hardware webcam & ONNX support:
```powershell
.\.venv\Scripts\pip.exe install -e ".[vision]"
```

### 2. Frontend
```powershell
cd apps\desktop
npm install
npm run dev
```
Open `http://localhost:5173` in Edge or Chrome for full Web Speech API voice input and spoken output.

### 3. Run Tests & Checks
```powershell
.\.venv\Scripts\pytest.exe -v
.\.venv\Scripts\ruff.exe check packages services tests
.\.venv\Scripts\mypy.exe packages services
```

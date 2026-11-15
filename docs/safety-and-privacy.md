# Safety, Responsible AI & Privacy Boundaries

## Responsible AI Guardrails

1. **Currency Non-Authentication (FR07)**: Vistara Vision provides visual denomination estimates and multi-note counting only. It **never** certifies that a banknote is genuine from ordinary RGB camera frames.
2. **Navigation Disclaimer**: Vistara Vision is an assistive information tool, not a certified navigation aid or obstacle-avoidance device.
3. **Honest Episodic Memory (FR11)**: Last-seen answers always report elapsed time, relative landmark position, source, and confidence. The system never claims an object is currently present from memory alone.
4. **Abstention Over Hallucination**: When lighting is too dark, glare is excessive, or confidence is below threshold, the system abstains and guides the user to reposition the camera.

## Privacy & Security

- **Local-First Processing**: Routine perception, embeddings, and memory are stored locally in `%LOCALAPPDATA%\VistaraVision\v1\vistara_memory.sqlite3`. Cloud vision is disabled by default.
- **No Always-On Recording**: Frames are processed on demand or in active sessions; bounded queues drop stale frames under load.
- **Phone Pairing Security (FR02/FR03)**: Pairing links use short-lived, single-use tokens with explicit camera consent on the phone. Expired, rejected, or disconnected tokens cannot be reused.
- **User Data Controls (FR13)**: Users can delete individual observations, delete learned objects, export memory, or clear all saved memory at any time.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  getApiBase,
  setApiBase,
  type CameraInfo,
  type DetectionItem,
  type DiagnosticsPayload,
  type LearnedObject,
  type ObservationItem,
  type PairingPayload
} from "./api";

declare global {
  interface Window {
    webkitSpeechRecognition?: new () => SpeechRecognitionLike;
    SpeechRecognition?: new () => SpeechRecognitionLike;
  }
}

interface SpeechRecognitionLike {
  lang: string;
  interimResults: boolean;
  maxAlternatives: number;
  start: () => void;
  stop: () => void;
  onresult: ((ev: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onerror: ((ev: { error: string }) => void) | null;
  onend: (() => void) | null;
}

export function App() {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>("webcam-0");
  const [cameraStatus, setCameraStatus] = useState<string>("stopped");
  const [announcement, setAnnouncement] = useState<string>(
    "Vistara Vision is ready. Press Start Vision or use the voice command bar."
  );
  const [autoSpeak, setAutoSpeak] = useState<boolean>(true);
  const [speechRate, setSpeechRate] = useState<number>(1.0);
  const [listening, setListening] = useState<boolean>(false);
  const [continuousMic, setContinuousMic] = useState<boolean>(false);
  const [lastHeard, setLastHeard] = useState<string>("");
  const [commandText, setCommandText] = useState<string>("");

  const [teachLabel, setTeachLabel] = useState<string>("");
  const [teachNote, setTeachNote] = useState<string>("");
  const [learnedObjects, setLearnedObjects] = useState<LearnedObject[]>([]);

  const [memoryQuery, setMemoryQuery] = useState<string>("");
  const [observations, setObservations] = useState<ObservationItem[]>([]);

  const [pairing, setPairing] = useState<PairingPayload | null>(null);
  const [showPreview, setShowPreview] = useState<boolean>(true);
  const [previewTick, setPreviewTick] = useState<number>(0);
  const [detections, setDetections] = useState<DetectionItem[]>([]);
  const [diagnostics, setDiagnostics] = useState<DiagnosticsPayload | null>(null);
  const [backendUrlInput, setBackendUrlInput] = useState<string>(() => getApiBase());
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const browserStreamRef = useRef<MediaStream | null>(null);
  const browserTimerRef = useRef<number | null>(null);
  const hiddenVideoRef = useRef<HTMLVideoElement | null>(null);

  const speak = useCallback(
    (text: string) => {
      setAnnouncement(text);
      if (!autoSpeak || typeof window === "undefined" || !("speechSynthesis" in window)) {
        return;
      }
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = speechRate;
      window.speechSynthesis.speak(utter);
    },
    [autoSpeak, speechRate]
  );

  const stopSpeaking = useCallback(() => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }, []);

  const refreshLists = useCallback(async () => {
    try {
      const [cams, objs, obs, diag] = await Promise.all([
        api.listCameras(),
        api.listObjects(),
        api.listObservations(),
        api.diagnostics()
      ]);
      setCameras(cams);
      setLearnedObjects(objs);
      setObservations(obs);
      setDiagnostics(diag);
      setCameraStatus(diag.camera_status);
    } catch (err) {
      speak(err instanceof Error ? err.message : "Could not reach local Vistara service.");
    }
  }, [speak]);

  useEffect(() => {
    void refreshLists();
  }, [refreshLists]);

  useEffect(() => {
    if (!showPreview || cameraStatus !== "active") return;
    const id = window.setInterval(() => setPreviewTick((t) => t + 1), 1000);
    return () => window.clearInterval(id);
  }, [showPreview, cameraStatus]);

  useEffect(() => {
    if (!pairing) return;
    const id = window.setInterval(async () => {
      try {
        const st = await api.pairingStatus(pairing.token);
        if (st.state === "connected") {
          setPairing((prev) => (prev ? { ...prev, state: "connected" } : null));
          setCameraStatus("active");
          speak("Phone camera connected and active.");
          void refreshLists();
          window.clearInterval(id);
        }
      } catch {
        window.clearInterval(id);
      }
    }, 1500);
    return () => window.clearInterval(id);
  }, [pairing, refreshLists, speak]);

  const stopBrowserStream = useCallback(() => {
    if (browserTimerRef.current !== null) {
      window.clearInterval(browserTimerRef.current);
      browserTimerRef.current = null;
    }
    if (browserStreamRef.current) {
      browserStreamRef.current.getTracks().forEach((t) => t.stop());
      browserStreamRef.current = null;
    }
  }, []);

  const startBrowserWebcam = useCallback(async (): Promise<boolean> => {
    if (!navigator.mediaDevices?.getUserMedia) {
      return false;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      browserStreamRef.current = stream;
      const video = hiddenVideoRef.current ?? document.createElement("video");
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      await video.play();
      hiddenVideoRef.current = video;

      const res = await api.startCamera("browser-cam-0");
      setSelectedCamera("browser-cam-0");
      setCameraStatus(res.status);

      const canvas = document.createElement("canvas");
      browserTimerRef.current = window.setInterval(() => {
        if (!video.videoWidth || !video.videoHeight) return;
        canvas.width = 640;
        canvas.height = Math.round((video.videoHeight / video.videoWidth) * 640) || 480;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(
          (blob) => {
            if (blob) void api.pushBrowserFrame(blob);
          },
          "image/jpeg",
          0.78
        );
      }, 500);

      speak("Browser webcam started.");
      await refreshLists();
      return true;
    } catch {
      stopBrowserStream();
      return false;
    }
  }, [refreshLists, speak, stopBrowserStream]);

  const handleStartCamera = async () => {
    stopBrowserStream();
    if (selectedCamera === "browser-cam-0") {
      const ok = await startBrowserWebcam();
      if (!ok) {
        const mockRes = await api.startCamera("mock-cam-0");
        setSelectedCamera("mock-cam-0");
        setCameraStatus(mockRes.status);
        speak("Browser camera was unavailable, so I started the Demo Scene Camera.");
        await refreshLists();
      }
      return;
    }

    try {
      const res = await api.startCamera(selectedCamera);
      setCameraStatus(res.status);
      speak(res.spoken_response);
      await refreshLists();
    } catch {
      if (selectedCamera === "webcam-0") {
        const browserOk = await startBrowserWebcam();
        if (browserOk) return;
        const mockRes = await api.startCamera("mock-cam-0");
        setSelectedCamera("mock-cam-0");
        setCameraStatus(mockRes.status);
        speak(
          "No physical webcam was accessible on this PC, so I started the Demo Scene Camera."
        );
        await refreshLists();
        return;
      }
      speak("Failed to start the selected camera.");
    }
  };

  const handleStopCamera = async () => {
    stopBrowserStream();
    try {
      const res = await api.stopCamera();
      setCameraStatus(res.status);
      speak(res.spoken_response);
      await refreshLists();
    } catch (err) {
      speak(err instanceof Error ? err.message : "Failed to stop camera.");
    }
  };

  const handleDescribe = useCallback(async () => {
    try {
      const res = await api.describeScene();
      setDetections(res.detections ?? []);
      setPreviewTick((t) => t + 1);
      speak(res.spoken_response);
      await refreshLists();
    } catch (err) {
      speak(err instanceof Error ? err.message : "Could not describe scene.");
    }
  }, [refreshLists, speak]);

  const handleReadText = async () => {
    try {
      const res = await api.readText();
      speak(res.spoken_response);
    } catch (err) {
      speak(err instanceof Error ? err.message : "Could not read text.");
    }
  };

  const handleCurrency = async () => {
    try {
      const res = await api.analyzeCurrency();
      speak(res.spoken_response);
    } catch (err) {
      speak(err instanceof Error ? err.message : "Could not analyze currency.");
    }
  };

  const handleTeach = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!teachLabel.trim()) {
      speak("Please enter a name for the object before teaching.");
      return;
    }
    try {
      const res = await api.teachObject(teachLabel.trim(), teachNote.trim() || undefined);
      setTeachLabel("");
      setTeachNote("");
      speak(res.spoken_response);
      await refreshLists();
    } catch (err) {
      speak(err instanceof Error ? err.message : "Could not learn object.");
    }
  };

  const handleAskMemory = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!memoryQuery.trim()) {
      speak("Enter an item name to search your memory.");
      return;
    }
    try {
      const res = await api.askLastSeen(memoryQuery.trim());
      speak(res.spoken_response);
    } catch (err) {
      speak(err instanceof Error ? err.message : "Memory search failed.");
    }
  };

  const handleCommandSubmit = useCallback(
    async (textToSend: string) => {
      if (!textToSend.trim()) return;
      try {
        const res = await api.sendCommand(textToSend.trim());
        setCommandText("");
        if (res.detections) {
          setDetections(res.detections);
        }
        setPreviewTick((t) => t + 1);
        speak(res.spoken_response);
        await refreshLists();
      } catch (err) {
        speak(err instanceof Error ? err.message : "Command failed.");
      }
    },
    [refreshLists, speak]
  );

  const startVoiceRecognition = useCallback(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      speak("Browser speech recognition is unavailable. Use Chrome or Edge, or type in the command box.");
      return;
    }
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
      setListening(false);
      return;
    }
    const rec = new SR();
    rec.lang = "en-IN";
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    recognitionRef.current = rec;
    setListening(true);

    rec.onresult = (ev) => {
      const transcript = ev.results[0]?.[0]?.transcript ?? "";
      if (transcript.trim()) {
        setLastHeard(transcript);
        setCommandText(transcript);
        void handleCommandSubmit(transcript);
      }
    };
    rec.onerror = (ev) => {
      setListening(false);
      recognitionRef.current = null;
      if (ev.error === "not-allowed") {
        setContinuousMic(false);
        speak("Microphone permission was denied. Please allow microphone access in your browser.");
      }
    };
    rec.onend = () => {
      setListening(false);
      recognitionRef.current = null;
      if (continuousMic) {
        window.setTimeout(() => {
          startVoiceRecognition();
        }, 400);
      }
    };
    rec.start();
  }, [continuousMic, handleCommandSubmit, speak]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (!e.altKey) return;
      const key = e.key.toLowerCase();
      if (key === "m") {
        e.preventDefault();
        startVoiceRecognition();
      } else if (key === "d") {
        e.preventDefault();
        void handleDescribe();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handleDescribe, startVoiceRecognition]);

  const handleCreatePairing = async () => {
    try {
      const res = await api.createPairing();
      setPairing(res);
      speak(res.spoken_response);
    } catch (err) {
      speak(err instanceof Error ? err.message : "Could not start phone pairing.");
    }
  };

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div className="brand">
          <h1>Vistara Vision</h1>
          <p>Voice-first assistive vision, currency reading &amp; honest memory</p>
        </div>
        <div className="status-strip">
          <span
            className={`badge ${cameraStatus === "active" ? "active" : ""}`}
            aria-label={`Camera status: ${cameraStatus}`}
          >
            Camera: {cameraStatus.toUpperCase()}
          </span>
          <button
            type="button"
            className="btn btn-danger"
            onClick={handleStopCamera}
            aria-label="Stop Camera immediately"
          >
            Stop Camera
          </button>
          <button type="button" className="btn" onClick={stopSpeaking}>
            Stop Speaking
          </button>
        </div>
      </header>

      <section className="announcement-banner" aria-live="assertive" aria-atomic="true">
        <p className="announcement-text">{announcement}</p>
        <button type="button" className="btn" onClick={() => speak(announcement)}>
          Repeat
        </button>
      </section>

      <form
        className="command-bar"
        onSubmit={(e) => {
          e.preventDefault();
          void handleCommandSubmit(commandText);
        }}
      >
        <label htmlFor="voice-cmd-input" className="muted" style={{ alignSelf: "center" }}>
          Command:
        </label>
        <input
          id="voice-cmd-input"
          type="text"
          value={commandText}
          onChange={(e) => setCommandText(e.target.value)}
          placeholder='Try: "What is in front of me?", "Count these notes", "Where was my bottle?"'
        />
        <button type="submit" className="btn btn-primary">
          Run Command
        </button>
        <button
          type="button"
          className={`btn ${listening ? "btn-mic-active" : ""}`}
          onClick={startVoiceRecognition}
          title="Shortcut: Alt+M"
        >
          {listening ? "Listening... (Click to Stop)" : "Speak (Alt+M)"}
        </button>
        <label style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
          <input
            type="checkbox"
            checked={continuousMic}
            onChange={(e) => {
              const checked = e.target.checked;
              setContinuousMic(checked);
              if (checked && !listening) {
                startVoiceRecognition();
              }
            }}
          />
          Hands-free
        </label>
      </form>
      {lastHeard && (
        <p className="muted" style={{ marginTop: "-0.75rem", marginBottom: "1rem" }}>
          Last heard: <strong>&ldquo;{lastHeard}&rdquo;</strong>
        </p>
      )}

      <main className="grid-4">
        {/* 1. Start Vision */}
        <section className="card" aria-labelledby="h-vision">
          <h2 id="h-vision">1. Start Vision</h2>
          <label htmlFor="camera-select" className="muted">
            Select camera source
          </label>
          <select
            id="camera-select"
            value={selectedCamera}
            onChange={(e) => setSelectedCamera(e.target.value)}
          >
            {cameras.map((cam) => (
              <option key={cam.source_id} value={cam.source_id}>
                {cam.name} {cam.available ? "" : "(Unavailable)"}
              </option>
            ))}
          </select>
          <div className="btn-row">
            <button type="button" className="btn btn-primary" onClick={handleStartCamera}>
              Start Vision
            </button>
            <button type="button" className="btn" onClick={handleDescribe}>
              What Do You See?
            </button>
            <button type="button" className="btn" onClick={handleReadText}>
              Read Text
            </button>
            <button type="button" className="btn" onClick={handleCurrency}>
              Identify Currency
            </button>
          </div>
          <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <input
              type="checkbox"
              checked={showPreview}
              onChange={(e) => setShowPreview(e.target.checked)}
            />
            Show visual preview (for sighted helper)
          </label>
          {showPreview && cameraStatus === "active" && (
            <div className="preview-container">
              <img
                className="preview-img"
                src={`${getApiBase()}/api/v1/cameras/preview?t=${previewTick}`}
                alt="Live camera preview showing what Vistara Vision sees"
              />
              <svg className="preview-overlay" viewBox="0 0 100 100" preserveAspectRatio="none">
                {detections.map((d, idx) => {
                  const x = d.box.x1 * 100;
                  const y = d.box.y1 * 100;
                  const w = Math.max(2, (d.box.x2 - d.box.x1) * 100);
                  const h = Math.max(2, (d.box.y2 - d.box.y1) * 100);
                  const color = d.is_personal_match ? "#22c55e" : "#38bdf8";
                  return (
                    <g key={`${d.label}-${idx}`}>
                      <rect
                        x={x}
                        y={y}
                        width={w}
                        height={h}
                        fill="none"
                        stroke={color}
                        strokeWidth="0.9"
                      />
                      <text
                        x={x + 1}
                        y={Math.max(5, y - 1.5)}
                        fill={color}
                        fontSize="4.2"
                        fontWeight="bold"
                      >
                        {d.label} ({Math.round(d.confidence * 100)}%)
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          )}
        </section>

        {/* 2. Connect Phone Camera */}
        <section className="card" aria-labelledby="h-phone">
          <h2 id="h-phone">2. Connect Phone Camera</h2>
          <p className="muted">
            Pair a phone browser with a single-use expiring link and explicit camera consent.
          </p>
          <button type="button" className="btn btn-primary" onClick={handleCreatePairing}>
            Generate QR &amp; Pairing Link
          </button>
          {pairing && (
            <div className="qr-box">
              <img src={pairing.qr_data_uri} alt="QR code to pair phone camera" />
              <strong>Code: {pairing.short_code}</strong>
              <span style={{ fontSize: "0.8rem", wordBreak: "break-all" }}>{pairing.join_url}</span>
              <span style={{ fontSize: "0.85rem" }}>State: {pairing.state}</span>
              <button
                type="button"
                className="btn btn-danger"
                onClick={async () => {
                  await api.disconnectPairing(pairing.token);
                  setPairing(null);
                  speak("Phone camera disconnected.");
                  void refreshLists();
                }}
              >
                Disconnect Phone
              </button>
            </div>
          )}
        </section>

        {/* 3. Teach Object */}
        <section className="card" aria-labelledby="h-teach">
          <h2 id="h-teach">3. Teach Object</h2>
          <form onSubmit={handleTeach} style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
            <label htmlFor="teach-label" className="muted">
              Object name
            </label>
            <input
              id="teach-label"
              type="text"
              value={teachLabel}
              onChange={(e) => setTeachLabel(e.target.value)}
              placeholder="e.g., Blue water bottle"
            />
            <input
              type="text"
              value={teachNote}
              onChange={(e) => setTeachNote(e.target.value)}
              placeholder="Optional note (e.g., Metal cap)"
              aria-label="Optional object note"
            />
            <button type="submit" className="btn btn-primary">
              Learn From Active Camera
            </button>
          </form>
          <ul className="list" aria-label="Learned personal objects">
            {learnedObjects.map((obj) => (
              <li key={obj.object_id} className="list-item">
                <span>
                  <strong>{obj.label}</strong>{" "}
                  <span className="muted">({obj.embedding_count} views)</span>
                </span>
                <button
                  type="button"
                  className="btn"
                  aria-label={`Delete ${obj.label}`}
                  onClick={async () => {
                    await api.deleteObject(obj.object_id);
                    speak(`Deleted ${obj.label}.`);
                    void refreshLists();
                  }}
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        </section>

        {/* 4. Ask Memory */}
        <section className="card" aria-labelledby="h-memory">
          <h2 id="h-memory">4. Ask Memory</h2>
          <form onSubmit={handleAskMemory} style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
            <label htmlFor="memory-query" className="muted">
              Ask where an item was last observed
            </label>
            <input
              id="memory-query"
              type="text"
              value={memoryQuery}
              onChange={(e) => setMemoryQuery(e.target.value)}
              placeholder="e.g., bottle, keys, charger"
            />
            <button type="submit" className="btn btn-primary">
              Where Was It?
            </button>
          </form>
          <ul className="list" aria-label="Recent observations">
            {observations.slice(0, 6).map((obs) => (
              <li key={obs.event_id} className="list-item">
                <span>
                  <strong>{obs.label}</strong> — <span className="muted">{obs.landmark_relation}</span>
                </span>
                <button
                  type="button"
                  className="btn"
                  aria-label={`Forget observation of ${obs.label}`}
                  onClick={async () => {
                    await api.deleteObservation(obs.event_id);
                    speak(`Forgot observation of ${obs.label}.`);
                    void refreshLists();
                  }}
                >
                  Forget
                </button>
              </li>
            ))}
          </ul>
          <button
            type="button"
            className="btn btn-danger"
            onClick={async () => {
              const res = await api.clearAllMemory();
              speak(res.spoken_response);
              void refreshLists();
            }}
          >
            Clear All Memory
          </button>
        </section>
      </main>

      <footer className="card" aria-labelledby="h-settings">
        <h2 id="h-settings">Accessibility &amp; Privacy Settings</h2>
        <div className="btn-row" style={{ alignItems: "center" }}>
          <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <input
              type="checkbox"
              checked={autoSpeak}
              onChange={(e) => setAutoSpeak(e.target.checked)}
            />
            Auto-speak responses
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            Speech speed ({speechRate.toFixed(1)}x)
            <input
              type="range"
              min={0.7}
              max={1.6}
              step={0.1}
              value={speechRate}
              onChange={(e) => setSpeechRate(Number(e.target.value))}
            />
          </label>
          {diagnostics && (
            <span className="muted">
              Local-only mode: {diagnostics.cloud_vision_enabled ? "Cloud enabled" : "On-device only"} |
              Learned objects: {diagnostics.learned_objects_count} | Dropped frames:{" "}
              {diagnostics.dropped_frames}
            </span>
          )}
        </div>
        <form
          className="btn-row"
          style={{ alignItems: "center", marginTop: "0.5rem" }}
          onSubmit={(e) => {
            e.preventDefault();
            const saved = setApiBase(backendUrlInput);
            setBackendUrlInput(saved);
            speak(`Backend URL set to ${saved || "same origin"}. Reconnecting.`);
            void refreshLists();
          }}
        >
          <label htmlFor="backend-url-input" className="muted">
            Backend URL:
          </label>
          <input
            id="backend-url-input"
            type="text"
            value={backendUrlInput}
            onChange={(e) => setBackendUrlInput(e.target.value)}
            placeholder="http://127.0.0.1:8765"
            style={{ minWidth: "240px" }}
          />
          <button type="submit" className="btn">
            Save &amp; Reconnect
          </button>
        </form>
      </footer>
    </div>
  );
}

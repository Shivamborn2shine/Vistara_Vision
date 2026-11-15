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
  continuous?: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start: () => void;
  stop: () => void;
  onresult:
    | ((ev: {
        results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal?: boolean }>;
      }) => void)
    | null;
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
  const [holdingToSpeak, setHoldingToSpeak] = useState<boolean>(false);
  const [micLevel, setMicLevel] = useState<number>(0);
  const [voiceStatus, setVoiceStatus] = useState<string>("Ready — press N or click Speak (N)");
  const [liveTranscript, setLiveTranscript] = useState<string>("");
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
  const pttActiveRef = useRef<boolean>(false);
  const pttTranscriptRef = useRef<string>("");
  const pttPendingSubmitRef = useRef<boolean>(false);
  const pttSubmitTimerRef = useRef<number | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const micCtxRef = useRef<AudioContext | null>(null);
  const micRafRef = useRef<number | null>(null);
  const continuousMicRef = useRef<boolean>(false);
  const isSpeakingRef = useRef<boolean>(false);
  const browserStreamRef = useRef<MediaStream | null>(null);
  const browserTimerRef = useRef<number | null>(null);
  const hiddenVideoRef = useRef<HTMLVideoElement | null>(null);
  const previewVideoRef = useRef<HTMLVideoElement | null>(null);
  const announcedPairingTokenRef = useRef<string | null>(null);
  const currencyAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    continuousMicRef.current = continuousMic;
  }, [continuousMic]);

  const speak = useCallback(
    (text: string) => {
      setAnnouncement(text);
      if (!autoSpeak || typeof window === "undefined" || !("speechSynthesis" in window)) {
        return;
      }
      if (recognitionRef.current) {
        recognitionRef.current.stop();
        recognitionRef.current = null;
        setListening(false);
      }
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = speechRate;
      isSpeakingRef.current = true;
      utter.onend = () => {
        isSpeakingRef.current = false;
        if (continuousMicRef.current && !recognitionRef.current) {
          window.setTimeout(() => startVoiceRecognition(), 350);
        }
      };
      utter.onerror = () => {
        isSpeakingRef.current = false;
      };
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
    if (!showPreview || cameraStatus !== "active") {
      setDetections([]);
      return;
    }
    if (previewVideoRef.current && browserStreamRef.current) {
      previewVideoRef.current.srcObject = browserStreamRef.current;
    }
    const id = window.setInterval(async () => {
      try {
        const res = await api.detectLive();
        setDetections(res.detections ?? []);
      } catch {
        // Ignore transient frame errors while camera warms up
      }
    }, 800);
    return () => window.clearInterval(id);
  }, [showPreview, cameraStatus, selectedCamera]);

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

  useEffect(() => {
    if (!pairing || pairing.state === "connected") return;
    const token = pairing.token;
    const id = window.setInterval(async () => {
      try {
        const st = await api.pairingStatus(token);
        if (st.state === "connected") {
          window.clearInterval(id);
          stopBrowserStream();
          setPairing((prev) => (prev ? { ...prev, state: "connected" } : null));
          if (st.camera_source_id) {
            setSelectedCamera(st.camera_source_id);
          }
          setCameraStatus("active");
          setPreviewTick((t) => t + 1);
          if (announcedPairingTokenRef.current !== token) {
            announcedPairingTokenRef.current = token;
            speak("Phone camera connected and active.");
          }
          void refreshLists();
        }
      } catch {
        window.clearInterval(id);
      }
    }, 1500);
    return () => window.clearInterval(id);
  }, [pairing?.token, pairing?.state, refreshLists, speak, stopBrowserStream]);

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
      }, 180);

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
    if (currencyAbortRef.current) {
      currencyAbortRef.current.abort();
    }
    stopSpeaking();
    const controller = new AbortController();
    currencyAbortRef.current = controller;
    setAnnouncement("Checking currency...");
    try {
      const res = await api.analyzeCurrency(controller.signal);
      if (currencyAbortRef.current === controller) {
        speak(res.spoken_response);
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return;
      }
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
    if (isSpeakingRef.current) {
      return;
    }
    const rec = new SR();
    rec.lang = "en-IN";
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    recognitionRef.current = rec;
    setListening(true);

    rec.onresult = (ev) => {
      if (isSpeakingRef.current || window.speechSynthesis?.speaking) {
        return;
      }
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
      if (continuousMicRef.current && !isSpeakingRef.current) {
        window.setTimeout(() => {
          if (continuousMicRef.current && !isSpeakingRef.current && !recognitionRef.current) {
            startVoiceRecognition();
          }
        }, 500);
      }
    };
    rec.start();
  }, [handleCommandSubmit, speak]);

  const playMicTone = useCallback((freq: number, durationMs: number) => {
    try {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      gain.gain.value = 0.08;
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + durationMs / 1000);
      window.setTimeout(() => {
        void ctx.close().catch(() => {});
      }, durationMs + 50);
    } catch {
      // Ignore audio cue errors if browser blocks AudioContext
    }
  }, []);

  const stopMicMeter = useCallback(() => {
    if (micRafRef.current !== null) {
      window.cancelAnimationFrame(micRafRef.current);
      micRafRef.current = null;
    }
    setMicLevel(0);
  }, []);

  const startMicMeter = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia) return;
      if (!micStreamRef.current) {
        micStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return;
      if (!micCtxRef.current || micCtxRef.current.state === "closed") {
        micCtxRef.current = new AudioCtx();
      }
      if (micCtxRef.current.state === "suspended") {
        await micCtxRef.current.resume();
      }
      const source = micCtxRef.current.createMediaStreamSource(micStreamRef.current);
      const analyser = micCtxRef.current.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      const data = new Uint8Array(analyser.frequencyBinCount);

      const tick = () => {
        if (!pttActiveRef.current) {
          setMicLevel(0);
          return;
        }
        analyser.getByteTimeDomainData(data);
        let sumSq = 0;
        for (let i = 0; i < data.length; i += 1) {
          const v = (data[i] - 128) / 128;
          sumSq += v * v;
        }
        const rms = Math.sqrt(sumSq / data.length);
        const level = Math.min(100, Math.round(rms * 320));
        setMicLevel(level);
        micRafRef.current = window.requestAnimationFrame(tick);
      };
      if (micRafRef.current !== null) {
        window.cancelAnimationFrame(micRafRef.current);
      }
      micRafRef.current = window.requestAnimationFrame(tick);
    } catch {
      setVoiceStatus("Microphone access failed — check browser mic permissions.");
    }
  }, []);

  useEffect(() => {
    return () => {
      if (micRafRef.current !== null) {
        window.cancelAnimationFrame(micRafRef.current);
      }
      if (micStreamRef.current) {
        for (const track of micStreamRef.current.getTracks()) {
          track.stop();
        }
        micStreamRef.current = null;
      }
      if (micCtxRef.current && micCtxRef.current.state !== "closed") {
        void micCtxRef.current.close().catch(() => {});
      }
    };
  }, []);

  const flushPttCommand = useCallback(() => {
    if (pttSubmitTimerRef.current !== null) {
      window.clearTimeout(pttSubmitTimerRef.current);
      pttSubmitTimerRef.current = null;
    }
    if (!pttPendingSubmitRef.current) return;
    pttPendingSubmitRef.current = false;
    const finalText = pttTranscriptRef.current.trim();
    pttTranscriptRef.current = "";
    if (finalText) {
      setLastHeard(finalText);
      setLiveTranscript(finalText);
      setCommandText(finalText);
      setVoiceStatus(`Sent: "${finalText}"`);
      void handleCommandSubmit(finalText);
    } else {
      setVoiceStatus("No words recognized — check the input level bar and speak closer.");
      setAnnouncement("No voice command heard while holding.");
    }
  }, [handleCommandSubmit]);

  const launchPttRecognition = useCallback(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      pttActiveRef.current = false;
      setHoldingToSpeak(false);
      stopMicMeter();
      const msg =
        "Browser speech recognition is unavailable. Use Chrome or Edge, or type in the command box.";
      setVoiceStatus(msg);
      speak(msg);
      return;
    }
    const rec = new SR();
    rec.lang = "en-IN";
    rec.continuous = true;
    rec.interimResults = true;
    rec.maxAlternatives = 1;
    recognitionRef.current = rec;
    setListening(true);
    setVoiceStatus("Listening... Speak now (press N or click to stop & send)");

    let sessionCommitted = pttTranscriptRef.current;

    rec.onresult = (ev) => {
      const parts: string[] = [];
      for (let i = 0; i < ev.results.length; i += 1) {
        const piece = ev.results[i]?.[0]?.transcript ?? "";
        if (piece.trim()) parts.push(piece.trim());
      }
      const combined = [sessionCommitted, parts.join(" ")].filter(Boolean).join(" ").trim();
      pttTranscriptRef.current = combined;
      if (combined) {
        setLastHeard(combined);
        setLiveTranscript(combined);
        setCommandText(combined);
      }
    };

    rec.onerror = (ev) => {
      const errCode = ev.error || "unknown";
      if (errCode === "no-speech") {
        setVoiceStatus("Listening... No speech detected yet.");
        return;
      }
      if (errCode === "aborted") {
        return;
      }
      pttActiveRef.current = false;
      pttPendingSubmitRef.current = false;
      setHoldingToSpeak(false);
      setListening(false);
      stopMicMeter();
      recognitionRef.current = null;
      if (errCode === "not-allowed") {
        const msg = "Microphone permission was denied. Allow mic access in browser settings.";
        setVoiceStatus(msg);
        speak(msg);
      } else if (errCode === "network" || errCode === "service-not-allowed") {
        const msg =
          "Speech recognition blocked by browser/network (Brave blocks speech servers). Use Chrome or Edge for voice commands.";
        setVoiceStatus(msg);
        setAnnouncement(msg);
      } else if (errCode === "audio-capture") {
        const msg = "No microphone found or microphone is busy.";
        setVoiceStatus(msg);
        speak(msg);
      } else {
        setVoiceStatus(`Speech error: ${errCode}`);
      }
    };

    rec.onend = () => {
      setListening(false);
      recognitionRef.current = null;
      if (pttActiveRef.current) {
        sessionCommitted = pttTranscriptRef.current;
        launchPttRecognition();
        return;
      }
      flushPttCommand();
    };

    try {
      rec.start();
    } catch {
      setVoiceStatus("Could not start speech recognition.");
    }
  }, [flushPttCommand, speak, stopMicMeter]);

  const startHoldToSpeak = useCallback(() => {
    if (pttActiveRef.current) return;
    stopSpeaking();
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    pttTranscriptRef.current = "";
    pttPendingSubmitRef.current = false;
    pttActiveRef.current = true;
    setHoldingToSpeak(true);
    setLiveTranscript("");
    setCommandText("");
    setVoiceStatus("Listening... Speak now");
    playMicTone(660, 80);
    void startMicMeter();
    launchPttRecognition();
  }, [launchPttRecognition, playMicTone, startMicMeter, stopSpeaking]);

  const stopHoldToSpeak = useCallback(() => {
    if (!pttActiveRef.current) return;
    pttActiveRef.current = false;
    pttPendingSubmitRef.current = true;
    setHoldingToSpeak(false);
    stopMicMeter();
    playMicTone(440, 80);
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      pttSubmitTimerRef.current = window.setTimeout(() => {
        flushPttCommand();
      }, 650);
    } else {
      flushPttCommand();
    }
  }, [flushPttCommand, playMicTone, stopMicMeter]);

  const toggleVoiceCapture = useCallback(() => {
    if (pttActiveRef.current) {
      stopHoldToSpeak();
    } else {
      startHoldToSpeak();
    }
  }, [startHoldToSpeak, stopHoldToSpeak]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isTypingField =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        Boolean(target?.isContentEditable);

      if (
        !isTypingField &&
        !e.ctrlKey &&
        !e.metaKey &&
        !e.altKey &&
        e.key.toLowerCase() === "n"
      ) {
        e.preventDefault();
        e.stopPropagation();
        if (!e.repeat) {
          toggleVoiceCapture();
        }
        return;
      }

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
    window.addEventListener("keydown", onKeyDown, true);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
    };
  }, [handleDescribe, startVoiceRecognition, toggleVoiceCapture]);

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

      <div className="hero-layout">
        <div>
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
              className={`btn ${holdingToSpeak ? "btn-hold-mic-active" : "btn-hold-mic"}`}
              onClick={toggleVoiceCapture}
              aria-pressed={holdingToSpeak}
              aria-label="Toggle voice command listening (Shortcut: N)"
              title="Press N or click to start listening; press N or click again to stop and send"
            >
              {holdingToSpeak ? "Listening... (Press N to Stop)" : "Speak (N)"}
            </button>
            <button
              type="button"
              className={`btn ${listening && !holdingToSpeak ? "btn-mic-active" : ""}`}
              onClick={startVoiceRecognition}
              title="Shortcut: Alt+M"
            >
              {listening && !holdingToSpeak ? "Listening... (Click to Stop)" : "Speak (Alt+M)"}
            </button>
            <label style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <input
                type="checkbox"
                checked={continuousMic}
                onChange={(e) => {
                  const checked = e.target.checked;
                  setContinuousMic(checked);
                  continuousMicRef.current = checked;
                  if (checked && !listening) {
                    startVoiceRecognition();
                  }
                }}
              />
              Hands-free
            </label>
          </form>
          <div
            className={`voice-monitor ${holdingToSpeak ? "voice-monitor-active" : ""}`}
            aria-live="polite"
          >
            <div className="voice-monitor-row">
              <span className="muted">
                <strong>Voice:</strong> {voiceStatus}
              </span>
              <div className="mic-meter-wrap" aria-label={`Microphone input level ${micLevel}%`}>
                <span className="muted" style={{ whiteSpace: "nowrap" }}>
                  Input: {micLevel}%
                </span>
                <div className="mic-meter-track">
                  <div className="mic-meter-fill" style={{ width: `${micLevel}%` }} />
                </div>
              </div>
            </div>
            <div className="muted">
              <strong>Live transcript:</strong>{" "}
              {liveTranscript
                ? `“${liveTranscript}”`
                : lastHeard
                  ? `Last heard: “${lastHeard}”`
                  : "Waiting for speech... Press N or click Speak (N) to start."}
            </div>
          </div>
        </div>

        <aside className="card" aria-label="Top-right live camera preview">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h2 style={{ fontSize: "1.1rem", margin: 0 }}>Live Preview</h2>
            <label style={{ display: "flex", alignItems: "center", gap: "0.35rem", fontSize: "0.9rem" }}>
              <input
                type="checkbox"
                checked={showPreview}
                onChange={(e) => setShowPreview(e.target.checked)}
              />
              Show
            </label>
          </div>
          {showPreview && cameraStatus === "active" ? (
            <div className="preview-container">
              {selectedCamera === "browser-cam-0" ? (
                <video
                  ref={previewVideoRef}
                  className="preview-img"
                  autoPlay
                  playsInline
                  muted
                />
              ) : (
                <img
                  className="preview-img"
                  src={`${getApiBase()}/api/v1/cameras/stream?t=${previewTick}`}
                  alt="Live camera preview showing what Vistara Vision sees"
                />
              )}
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
          ) : (
            <div
              className="preview-container"
              style={{
                height: "180px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                padding: "1rem",
                textAlign: "center"
              }}
            >
              <span className="muted">
                {cameraStatus === "active" ? "Preview hidden." : "Start Vision to see live camera feed."}
              </span>
            </div>
          )}
        </aside>
      </div>

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

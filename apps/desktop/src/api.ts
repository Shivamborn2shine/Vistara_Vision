export interface CameraInfo {
  source_id: string;
  name: string;
  source_type: string;
  available: boolean;
}

export interface LearnedObject {
  object_id: string;
  label: string;
  aliases: string[];
  owner_note: string | null;
  embedding_count: number;
  created_at: string;
  edited_at: string;
}

export interface ObservationItem {
  event_id: string;
  object_id: string | null;
  label: string;
  label_source: string;
  timestamp_utc: string;
  camera_id: string;
  room_id: string | null;
  landmark_relation: string;
  confidence: number;
}

export interface PairingPayload {
  token: string;
  short_code: string;
  join_url: string;
  qr_data_uri: string;
  expires_at: string;
  state: string;
  spoken_response: string;
}

export interface DiagnosticsPayload {
  uptime_seconds: number;
  camera_id: string | null;
  camera_status: string;
  dropped_frames: number;
  cloud_vision_enabled: boolean;
  models: Record<string, string>;
  learned_objects_count: number;
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data as T;
}

export const api = {
  listCameras: () => request<CameraInfo[]>("/api/v1/cameras"),
  startCamera: (sourceId?: string) =>
    request<{ camera_id: string; status: string; spoken_response: string }>("/api/v1/cameras/start", {
      method: "POST",
      body: JSON.stringify({ source_id: sourceId ?? null })
    }),
  stopCamera: () =>
    request<{ camera_id: string | null; status: string; spoken_response: string }>(
      "/api/v1/cameras/stop",
      { method: "POST" }
    ),
  describeScene: () =>
    request<{ spoken_response: string; detections: unknown[] }>("/api/v1/vision/describe", {
      method: "POST"
    }),
  readText: () =>
    request<{ spoken_response: string; full_text: string }>("/api/v1/vision/ocr", {
      method: "POST"
    }),
  analyzeCurrency: () =>
    request<{ spoken_response: string; total_inr: number; distinct_count: number; abstained: boolean }>(
      "/api/v1/vision/currency",
      { method: "POST" }
    ),
  listObjects: () => request<LearnedObject[]>("/api/v1/objects"),
  teachObject: (label: string, ownerNote?: string) =>
    request<{ object_id: string; label: string; spoken_response: string }>("/api/v1/objects/teach", {
      method: "POST",
      body: JSON.stringify({ label, owner_note: ownerNote || null })
    }),
  deleteObject: (objectId: string) =>
    request<{ deleted: boolean }>(`/api/v1/objects/${objectId}`, { method: "DELETE" }),
  listObservations: () => request<ObservationItem[]>("/api/v1/memory/observations"),
  askLastSeen: (item: string) =>
    request<{ found: boolean; spoken_response: string }>(
      `/api/v1/memory/last-seen?item=${encodeURIComponent(item)}`
    ),
  deleteObservation: (eventId: string) =>
    request<{ deleted: boolean }>(`/api/v1/memory/observations/${eventId}`, { method: "DELETE" }),
  clearAllMemory: () =>
    request<{ cleared: boolean; spoken_response: string }>("/api/v1/memory", { method: "DELETE" }),
  sendCommand: (text: string) =>
    request<{ intent: string; spoken_response: string }>("/api/v1/commands", {
      method: "POST",
      body: JSON.stringify({ text })
    }),
  createPairing: () => request<PairingPayload>("/api/v1/pairing/create", { method: "POST" }),
  pairingStatus: (token: string) =>
    request<{ state: string; camera_source_id: string | null }>(`/api/v1/pairing/status/${token}`),
  disconnectPairing: (token: string) =>
    request<{ disconnected: boolean }>(`/api/v1/pairing/disconnect/${token}`, { method: "POST" }),
  diagnostics: () => request<DiagnosticsPayload>("/api/v1/diagnostics")
};

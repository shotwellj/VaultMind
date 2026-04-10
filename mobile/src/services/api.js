/**
 * VaultMind API Service
 *
 * Connects the mobile app to the VaultMind backend server.
 * All communication goes over WireGuard VPN or local network.
 * No cloud relay. No intermediary servers.
 */

import * as SecureStore from "expo-secure-store";

const SERVER_KEY = "vaultmind_server_url";
const TOKEN_KEY = "vaultmind_auth_token";
const DEVICE_KEY = "vaultmind_device_id";

let _serverUrl = null;
let _authToken = null;
let _deviceId = null;

// -- Connection Setup --

export async function configureServer(url) {
  _serverUrl = url.replace(/\/$/, "");
  await SecureStore.setItemAsync(SERVER_KEY, _serverUrl);
}

export async function getServerUrl() {
  if (!_serverUrl) {
    _serverUrl = await SecureStore.getItemAsync(SERVER_KEY);
  }
  return _serverUrl;
}

export async function setAuthToken(token) {
  _authToken = token;
  await SecureStore.setItemAsync(TOKEN_KEY, token);
}

export async function getAuthToken() {
  if (!_authToken) {
    _authToken = await SecureStore.getItemAsync(TOKEN_KEY);
  }
  return _authToken;
}

export async function getDeviceId() {
  if (!_deviceId) {
    _deviceId = await SecureStore.getItemAsync(DEVICE_KEY);
    if (!_deviceId) {
      _deviceId = `mobile-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      await SecureStore.setItemAsync(DEVICE_KEY, _deviceId);
    }
  }
  return _deviceId;
}

// -- Base Request --

async function request(path, options = {}) {
  const url = await getServerUrl();
  if (!url) throw new Error("Server not configured. Go to Settings.");

  const token = await getAuthToken();
  const headers = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...options.headers,
  };

  const response = await fetch(`${url}${path}`, {
    ...options,
    headers,
    timeout: options.timeout || 30000,
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error");
    throw new Error(`API ${response.status}: ${text}`);
  }

  return response;
}

async function get(path) {
  const res = await request(path);
  return res.json();
}

async function post(path, body) {
  const res = await request(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
  return res.json();
}

// -- Chat API --

export async function streamChat(message, model, mode, history, onToken, onStatus, onDone) {
  const url = await getServerUrl();
  const token = await getAuthToken();

  const response = await fetch(`${url}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ message, model, mode, history }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      try {
        const data = JSON.parse(line.slice(6));
        if (data.token) onToken?.(data.token);
        else if (data.status) onStatus?.(data.status);
        else if (data.done) onDone?.(data);
        else if (data.quality) onStatus?.(`Quality: ${data.quality.badge_text}`);
      } catch {}
    }
  }
}

// -- Vault API --

export async function listFiles() {
  return get("/files");
}

export async function searchVault(query) {
  return post("/chat", { message: query, mode: "vault" });
}

// -- Sync API --

export async function syncPull(lastSyncOffset) {
  const deviceId = await getDeviceId();
  return post("/sync/pull", {
    device_id: deviceId,
    last_sync_offset: lastSyncOffset || null,
  });
}

export async function syncPush(items) {
  const deviceId = await getDeviceId();
  return post("/sync/push", {
    device_id: deviceId,
    items,
  });
}

export async function registerDevice() {
  const deviceId = await getDeviceId();
  return post("/sync/register", {
    device_id: deviceId,
    device_type: "mobile",
    platform: "react-native",
  });
}

// -- Photo Pipeline API --

export async function uploadPhoto(base64Data, filename, docType) {
  return post("/photos/process", {
    image_data: base64Data,
    filename: filename || `photo-${Date.now()}.jpg`,
    document_type: docType || "auto",
  });
}

// -- Call Intelligence API --

export async function processTranscript(transcript, metadata) {
  return post("/calls/process", {
    transcript,
    ...metadata,
  });
}

// -- Alerts API --

export async function getPendingAlerts() {
  const deviceId = await getDeviceId();
  return get(`/proactive/alerts?device_id=${deviceId}`);
}

export async function getUnreadCount() {
  return get("/proactive/unread");
}

export async function markAlertRead(alertId) {
  return post(`/proactive/alerts/${alertId}/read`, {});
}

// -- Contact Intelligence API --

export async function getContactBriefing(contactId) {
  return get(`/contacts/${contactId}/briefing`);
}

export async function searchContacts(query) {
  return get(`/contacts/search?q=${encodeURIComponent(query)}`);
}

export async function importContacts(contacts) {
  return post("/contacts/import", { contacts });
}

// -- Health Check --

export async function checkConnection() {
  try {
    const url = await getServerUrl();
    if (!url) return { connected: false, reason: "No server configured" };
    const res = await fetch(`${url}/models`, { timeout: 5000 });
    return { connected: res.ok, latency: "ok" };
  } catch (e) {
    return { connected: false, reason: e.message };
  }
}

/**
 * VaultMind Sync Service
 *
 * Handles incremental sync between the mobile app and the VaultMind server.
 * Uses the secure sync protocol (WireGuard + HMAC-SHA256 signing).
 *
 * Key principles:
 * - Sync only changed items (incremental via offsets)
 * - Resume interrupted syncs (track last offset)
 * - Cache recent conversations locally for offline access
 * - Queue outbound items when offline (photos, transcripts, notes)
 */

import * as SecureStore from "expo-secure-store";
import * as FileSystem from "expo-file-system";
import { syncPull, syncPush, registerDevice } from "./api";

const SYNC_OFFSET_KEY = "vaultmind_sync_offset";
const SYNC_QUEUE_DIR = `${FileSystem.documentDirectory}sync_queue/`;
const CACHE_DIR = `${FileSystem.documentDirectory}cache/`;

let _syncInProgress = false;

// -- Initialization --

export async function initSync() {
  await FileSystem.makeDirectoryAsync(SYNC_QUEUE_DIR, { intermediates: true }).catch(() => {});
  await FileSystem.makeDirectoryAsync(CACHE_DIR, { intermediates: true }).catch(() => {});

  try {
    await registerDevice();
  } catch (e) {
    console.warn("[Sync] Device registration failed:", e.message);
  }
}

// -- Pull (server -> phone) --

export async function pullFromServer() {
  if (_syncInProgress) return { skipped: true };
  _syncInProgress = true;

  try {
    const lastOffset = await SecureStore.getItemAsync(SYNC_OFFSET_KEY);
    const result = await syncPull(lastOffset);

    if (result.items && result.items.length > 0) {
      // Cache pulled items locally
      for (const item of result.items) {
        const cachePath = `${CACHE_DIR}${item.id || Date.now()}.json`;
        await FileSystem.writeAsStringAsync(cachePath, JSON.stringify(item));
      }
    }

    // Update sync offset
    if (result.new_offset) {
      await SecureStore.setItemAsync(SYNC_OFFSET_KEY, String(result.new_offset));
    }

    return {
      pulled: result.items?.length || 0,
      offset: result.new_offset,
    };
  } catch (e) {
    console.warn("[Sync] Pull failed:", e.message);
    return { error: e.message };
  } finally {
    _syncInProgress = false;
  }
}

// -- Push (phone -> server) --

export async function pushToServer() {
  if (_syncInProgress) return { skipped: true };
  _syncInProgress = true;

  try {
    // Read queued items
    const files = await FileSystem.readDirectoryAsync(SYNC_QUEUE_DIR).catch(() => []);
    if (files.length === 0) return { pushed: 0 };

    const items = [];
    for (const file of files) {
      const content = await FileSystem.readAsStringAsync(`${SYNC_QUEUE_DIR}${file}`);
      items.push(JSON.parse(content));
    }

    // Push to server
    const result = await syncPush(items);

    // Remove pushed items from queue
    if (result.status === "ok" || result.accepted) {
      for (const file of files) {
        await FileSystem.deleteAsync(`${SYNC_QUEUE_DIR}${file}`, { idempotent: true });
      }
    }

    return { pushed: items.length };
  } catch (e) {
    console.warn("[Sync] Push failed (items queued for retry):", e.message);
    return { error: e.message };
  } finally {
    _syncInProgress = false;
  }
}

// -- Queue for Offline --

export async function queueItem(item) {
  const filename = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}.json`;
  await FileSystem.writeAsStringAsync(
    `${SYNC_QUEUE_DIR}${filename}`,
    JSON.stringify({ ...item, queued_at: new Date().toISOString() })
  );
}

export async function getQueueSize() {
  const files = await FileSystem.readDirectoryAsync(SYNC_QUEUE_DIR).catch(() => []);
  return files.length;
}

// -- Full Sync Cycle --

export async function runSyncCycle() {
  const pushResult = await pushToServer();
  const pullResult = await pullFromServer();
  return { push: pushResult, pull: pullResult };
}

// -- Cached Data Access (offline) --

export async function getCachedConversations(limit = 20) {
  try {
    const files = await FileSystem.readDirectoryAsync(CACHE_DIR);
    const items = [];
    const sorted = files.sort().reverse().slice(0, limit);

    for (const file of sorted) {
      const content = await FileSystem.readAsStringAsync(`${CACHE_DIR}${file}`);
      items.push(JSON.parse(content));
    }
    return items;
  } catch {
    return [];
  }
}

export async function clearCache() {
  await FileSystem.deleteAsync(CACHE_DIR, { idempotent: true });
  await FileSystem.makeDirectoryAsync(CACHE_DIR, { intermediates: true });
}

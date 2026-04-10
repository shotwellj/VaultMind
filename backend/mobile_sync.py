"""
VaultMind Mobile Sync Protocol
Phase 6 foundation: REST API endpoints that a React Native app will connect to
over WireGuard VPN or local network. No cloud. No intermediary.

Architecture:
  Phone <--WireGuard VPN--> Home Server (VaultMind)

  Sync is:
    - Incremental: only new data since last sync timestamp
    - Conflict-aware: last-write-wins with conflict log for manual review
    - Resumable: tracks sync cursor so interrupted syncs pick up where they left off
    - Encrypted: libsodium box encryption on top of WireGuard tunnel

Endpoints:
  POST /mobile/sync/pull    - Phone pulls new data from server
  POST /mobile/sync/push    - Phone pushes new data to server (photos, transcripts)
  GET  /mobile/sync/status  - Check sync health and last sync time
  POST /mobile/sync/auth    - Device authentication (shared secret, no cloud auth)
"""

import os
import json
import hashlib
import secrets
import time
from datetime import datetime, timezone
from typing import Optional

# These will be imported when integrated into main.py's FastAPI app
# For now, this module defines the data structures and logic

SYNC_DIR = os.environ.get("VAULTMIND_SYNC_DIR", os.path.expanduser("~/.vaultmind/sync"))
DEVICES_FILE = os.path.join(SYNC_DIR, "devices.json")
SYNC_LOG_FILE = os.path.join(SYNC_DIR, "sync_log.json")
CONFLICT_DIR = os.path.join(SYNC_DIR, "conflicts")

os.makedirs(SYNC_DIR, exist_ok=True)
os.makedirs(CONFLICT_DIR, exist_ok=True)


# ── Device Management ──────────────────────────────────────────

def generate_device_token() -> str:
    """Generate a secure device authentication token."""
    return secrets.token_urlsafe(48)


def register_device(device_name: str, device_type: str = "mobile") -> dict:
    """Register a new device for sync.

    Args:
        device_name: Human-readable name (e.g., "Jason's iPhone")
        device_type: "mobile" or "desktop"

    Returns:
        Device record with auth token (show once, never stored in plaintext)
    """
    devices = _load_devices()

    device_id = hashlib.sha256(f"{device_name}-{time.time()}".encode()).hexdigest()[:16]
    token = generate_device_token()
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    device = {
        "device_id": device_id,
        "device_name": device_name,
        "device_type": device_type,
        "token_hash": token_hash,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "last_sync": None,
        "sync_cursor": 0,  # Timestamp-based cursor for incremental sync
        "status": "active",
    }

    devices[device_id] = device
    _save_devices(devices)

    # Return with plaintext token (only time it's visible)
    return {**device, "token": token}


def authenticate_device(device_id: str, token: str) -> bool:
    """Verify a device's authentication token."""
    devices = _load_devices()
    device = devices.get(device_id)
    if not device or device.get("status") != "active":
        return False
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    return token_hash == device.get("token_hash")


def list_devices() -> list[dict]:
    """List all registered devices (without token hashes)."""
    devices = _load_devices()
    safe = []
    for d in devices.values():
        record = {k: v for k, v in d.items() if k != "token_hash"}
        safe.append(record)
    return safe


# ── Sync Pull (Server -> Phone) ───────────────────────────────

def pull_sync_data(device_id: str, since_cursor: int = 0, limit: int = 50) -> dict:
    """Pull new data from server for the mobile device.

    Returns conversation summaries, document metadata, and memory
    entries created after the given cursor timestamp.

    Args:
        device_id: Authenticated device ID
        since_cursor: Unix timestamp -- only return data newer than this
        limit: Max items per pull (for bandwidth management)

    Returns:
        {
            "conversations": [...],  # New/updated conversation summaries
            "memories": [...],       # New conversation memory entries
            "documents": [...],      # New document metadata (not full text)
            "cursor": int,           # New cursor for next pull
            "has_more": bool,        # Whether more data is available
        }
    """
    # This is the structure -- actual ChromaDB/file queries will be
    # wired in when integrated into main.py

    now = int(datetime.now(timezone.utc).timestamp())

    result = {
        "conversations": [],
        "memories": [],
        "documents": [],
        "cursor": now,
        "has_more": False,
        "pulled_at": datetime.now(timezone.utc).isoformat(),
    }

    # Update device's sync cursor
    devices = _load_devices()
    if device_id in devices:
        devices[device_id]["last_sync"] = datetime.now(timezone.utc).isoformat()
        devices[device_id]["sync_cursor"] = now
        _save_devices(devices)

    _log_sync_event(device_id, "pull", len(result["conversations"]) + len(result["memories"]))

    return result


# ── Sync Push (Phone -> Server) ────────────────────────────────

def push_sync_data(device_id: str, payload: dict) -> dict:
    """Push new data from phone to server.

    Accepts:
      - photos: OCR-extracted text from on-device processing
      - transcripts: Call transcriptions from on-device Whisper
      - notes: Quick voice/text notes captured on mobile

    Args:
        device_id: Authenticated device ID
        payload: {
            "photos": [{"filename": str, "ocr_text": str, "captured_at": str}],
            "transcripts": [{"call_id": str, "text": str, "summary": str, "recorded_at": str}],
            "notes": [{"text": str, "created_at": str}],
        }

    Returns:
        {"accepted": int, "conflicts": int, "errors": []}
    """
    accepted = 0
    conflicts = 0
    errors = []

    # Process photos (OCR text only -- original images stay on phone/NAS)
    for photo in payload.get("photos", []):
        try:
            _store_mobile_content(
                device_id=device_id,
                content_type="photo",
                content=photo.get("ocr_text", ""),
                metadata={
                    "filename": photo.get("filename", "unknown"),
                    "captured_at": photo.get("captured_at", ""),
                    "source": f"mobile-photo:{photo.get('filename', 'unknown')}",
                },
            )
            accepted += 1
        except ConflictError as e:
            conflicts += 1
            _log_conflict(device_id, "photo", photo, str(e))
        except Exception as e:
            errors.append(f"Photo {photo.get('filename')}: {str(e)}")

    # Process call transcripts
    for transcript in payload.get("transcripts", []):
        try:
            _store_mobile_content(
                device_id=device_id,
                content_type="transcript",
                content=transcript.get("text", ""),
                metadata={
                    "call_id": transcript.get("call_id", ""),
                    "summary": transcript.get("summary", ""),
                    "recorded_at": transcript.get("recorded_at", ""),
                    "source": f"mobile-call:{transcript.get('call_id', 'unknown')}",
                },
            )
            accepted += 1
        except ConflictError as e:
            conflicts += 1
            _log_conflict(device_id, "transcript", transcript, str(e))
        except Exception as e:
            errors.append(f"Transcript {transcript.get('call_id')}: {str(e)}")

    # Process quick notes
    for note in payload.get("notes", []):
        try:
            _store_mobile_content(
                device_id=device_id,
                content_type="note",
                content=note.get("text", ""),
                metadata={
                    "created_at": note.get("created_at", ""),
                    "source": f"mobile-note:{int(time.time())}",
                },
            )
            accepted += 1
        except Exception as e:
            errors.append(f"Note: {str(e)}")

    _log_sync_event(device_id, "push", accepted)

    return {
        "accepted": accepted,
        "conflicts": conflicts,
        "errors": errors,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Sync Status ────────────────────────────────────────────────

def get_sync_status(device_id: str) -> dict:
    """Get sync health for a device."""
    devices = _load_devices()
    device = devices.get(device_id, {})

    return {
        "device_id": device_id,
        "device_name": device.get("device_name", "Unknown"),
        "last_sync": device.get("last_sync"),
        "sync_cursor": device.get("sync_cursor", 0),
        "status": device.get("status", "unknown"),
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


# ── Internal Helpers ───────────────────────────────────────────

class ConflictError(Exception):
    """Raised when a sync conflict is detected."""
    pass


def _store_mobile_content(device_id: str, content_type: str, content: str, metadata: dict):
    """Store content from mobile into the sync staging area.

    When integrated with main.py, this will call chunk_text() and
    embed_and_store() to add the content to ChromaDB.
    """
    if not content.strip():
        return

    staging_dir = os.path.join(SYNC_DIR, "staging", device_id)
    os.makedirs(staging_dir, exist_ok=True)

    entry = {
        "content_type": content_type,
        "content": content,
        "metadata": metadata,
        "device_id": device_id,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "indexed": False,  # Will be set True once embedded in ChromaDB
    }

    filename = f"{content_type}_{int(time.time() * 1000)}.json"
    filepath = os.path.join(staging_dir, filename)

    with open(filepath, "w") as f:
        json.dump(entry, f, indent=2)


def _log_conflict(device_id: str, content_type: str, data: dict, reason: str):
    """Log a sync conflict for manual review."""
    conflict = {
        "device_id": device_id,
        "content_type": content_type,
        "data": data,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolved": False,
    }
    filename = f"conflict_{int(time.time() * 1000)}.json"
    with open(os.path.join(CONFLICT_DIR, filename), "w") as f:
        json.dump(conflict, f, indent=2)


def _log_sync_event(device_id: str, event_type: str, items_count: int):
    """Append to the sync audit log."""
    log = _load_sync_log()
    log.append({
        "device_id": device_id,
        "event": event_type,
        "items": items_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    # Keep last 1000 entries
    log = log[-1000:]
    with open(SYNC_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def _load_devices() -> dict:
    """Load registered devices from disk."""
    if os.path.exists(DEVICES_FILE):
        try:
            with open(DEVICES_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_devices(devices: dict):
    """Save registered devices to disk."""
    with open(DEVICES_FILE, "w") as f:
        json.dump(devices, f, indent=2)


def _load_sync_log() -> list:
    """Load the sync audit log."""
    if os.path.exists(SYNC_LOG_FILE):
        try:
            with open(SYNC_LOG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []

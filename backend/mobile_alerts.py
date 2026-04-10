"""
Mobile Push Alerts for VaultMind

Local push notification server for mobile devices:
  - Alert queue per device (SQLite storage)
  - Priority filtering: urgent, high, medium, low
  - Integration with proactive_intel.py alerts
  - Device registration with push preferences
  - Pull-based: mobile app polls /alerts/pending endpoint
  - Alert delivery tracking
  - Quiet hours support
  - Batch digest delivery

Storage: ~/.vaultmind/alerts/
"""

import os
import json
import sqlite3
import hashlib
import time
from datetime import datetime, timezone, time as dt_time
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from enum import Enum

ALERTS_DIR = os.environ.get("VAULTMIND_ALERTS_DIR", os.path.expanduser("~/.vaultmind/alerts"))
ALERTS_DB = os.path.join(ALERTS_DIR, "mobile_alerts.db")

os.makedirs(ALERTS_DIR, exist_ok=True)


class AlertPriority(str, Enum):
    """Alert priority levels."""
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertStatus(str, Enum):
    """Alert delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    DISMISSED = "dismissed"


@dataclass
class MobileDevice:
    """Mobile device registration."""
    device_id: str
    device_name: str
    device_type: str
    min_priority: str = "medium"
    enabled: bool = True
    last_poll: Optional[str] = None
    registered_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AlertPreferences:
    """Alert preferences per device."""
    device_id: str
    min_priority: str = "medium"
    quiet_hours_start: Optional[str] = None  # HH:MM format
    quiet_hours_end: Optional[str] = None
    digest_enabled: bool = False
    digest_hour: int = 9  # 9 AM
    enabled: bool = True
    updated_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PushAlert:
    """Push alert for mobile."""
    alert_id: str
    device_id: str
    title: str
    message: str
    priority: str
    alert_type: str
    source: str = ""
    status: str = "pending"
    created_at: Optional[str] = None
    sent_at: Optional[str] = None
    delivered_at: Optional[str] = None
    read_at: Optional[str] = None
    data: Dict = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.data is None:
            self.data = {}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# -- Database Initialization --

def _init_db():
    """Initialize alert database schema."""
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()

    # Devices table
    c.execute('''CREATE TABLE IF NOT EXISTS devices (
        device_id TEXT PRIMARY KEY,
        device_name TEXT NOT NULL,
        device_type TEXT,
        min_priority TEXT DEFAULT "medium",
        enabled BOOLEAN DEFAULT 1,
        last_poll TEXT,
        registered_at TEXT
    )''')

    # Alert preferences table
    c.execute('''CREATE TABLE IF NOT EXISTS preferences (
        device_id TEXT PRIMARY KEY,
        min_priority TEXT DEFAULT "medium",
        quiet_hours_start TEXT,
        quiet_hours_end TEXT,
        digest_enabled BOOLEAN DEFAULT 0,
        digest_hour INTEGER DEFAULT 9,
        enabled BOOLEAN DEFAULT 1,
        updated_at TEXT
    )''')

    # Push alerts table
    c.execute('''CREATE TABLE IF NOT EXISTS alerts (
        alert_id TEXT PRIMARY KEY,
        device_id TEXT NOT NULL,
        title TEXT NOT NULL,
        message TEXT,
        priority TEXT,
        alert_type TEXT,
        source TEXT,
        status TEXT DEFAULT "pending",
        created_at TEXT,
        sent_at TEXT,
        delivered_at TEXT,
        read_at TEXT,
        data TEXT,
        FOREIGN KEY(device_id) REFERENCES devices(device_id)
    )''')

    # Alert digest records
    c.execute('''CREATE TABLE IF NOT EXISTS digests (
        digest_id TEXT PRIMARY KEY,
        device_id TEXT NOT NULL,
        alert_ids TEXT,
        sent_at TEXT,
        FOREIGN KEY(device_id) REFERENCES devices(device_id)
    )''')

    conn.commit()
    conn.close()


_init_db()


# -- Device Management --

def register_device(device_id: str, device_name: str, device_type: str = "mobile") -> MobileDevice:
    """Register a new mobile device.

    Args:
        device_id: Device identifier
        device_name: Human-readable name
        device_type: "mobile" or "tablet"

    Returns:
        MobileDevice record
    """
    now = datetime.now(timezone.utc).isoformat()

    device = MobileDevice(
        device_id=device_id,
        device_name=device_name,
        device_type=device_type,
        registered_at=now,
    )

    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO devices (device_id, device_name, device_type, registered_at)
                 VALUES (?, ?, ?, ?)''',
              (device_id, device_name, device_type, now))
    conn.commit()
    conn.close()

    return device


def get_device(device_id: str) -> Optional[MobileDevice]:
    """Get device record."""
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('SELECT * FROM devices WHERE device_id = ?', (device_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return MobileDevice(
            device_id=row[0],
            device_name=row[1],
            device_type=row[2],
            min_priority=row[3],
            enabled=bool(row[4]),
            last_poll=row[5],
            registered_at=row[6],
        )
    return None


def list_devices() -> List[MobileDevice]:
    """List all registered devices."""
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('SELECT * FROM devices')
    rows = c.fetchall()
    conn.close()

    return [MobileDevice(
        device_id=row[0],
        device_name=row[1],
        device_type=row[2],
        min_priority=row[3],
        enabled=bool(row[4]),
        last_poll=row[5],
        registered_at=row[6],
    ) for row in rows]


# -- Preferences --

def configure_preferences(device_id: str, min_priority: str = "medium",
                          quiet_hours_start: Optional[str] = None,
                          quiet_hours_end: Optional[str] = None,
                          digest_enabled: bool = False) -> AlertPreferences:
    """Configure alert preferences for a device.

    Args:
        device_id: Device identifier
        min_priority: Minimum priority to receive ("urgent", "high", "medium", "low")
        quiet_hours_start: Start of quiet hours (HH:MM)
        quiet_hours_end: End of quiet hours (HH:MM)
        digest_enabled: Enable digest mode

    Returns:
        AlertPreferences record
    """
    now = datetime.now(timezone.utc).isoformat()

    prefs = AlertPreferences(
        device_id=device_id,
        min_priority=min_priority,
        quiet_hours_start=quiet_hours_start,
        quiet_hours_end=quiet_hours_end,
        digest_enabled=digest_enabled,
        updated_at=now,
    )

    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO preferences
                 (device_id, min_priority, quiet_hours_start, quiet_hours_end, digest_enabled, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (device_id, min_priority, quiet_hours_start, quiet_hours_end, int(digest_enabled), now))
    conn.commit()
    conn.close()

    return prefs


def get_preferences(device_id: str) -> AlertPreferences:
    """Get alert preferences for device."""
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('SELECT * FROM preferences WHERE device_id = ?', (device_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return AlertPreferences(
            device_id=row[0],
            min_priority=row[1],
            quiet_hours_start=row[2],
            quiet_hours_end=row[3],
            digest_enabled=bool(row[4]),
            digest_hour=row[5],
            enabled=bool(row[6]),
            updated_at=row[7],
        )

    # Return defaults
    return AlertPreferences(device_id=device_id)


# -- Alert Queueing --

def queue_alert_for_device(device_id: str, title: str, message: str,
                            priority: str = "medium", alert_type: str = "general",
                            source: str = "", data: Optional[dict] = None) -> PushAlert:
    """Queue an alert for a specific device.

    Args:
        device_id: Target device
        title: Alert title
        message: Alert message
        priority: Alert priority
        alert_type: Type of alert
        source: Source (file path, URL, etc.)
        data: Additional data dict

    Returns:
        PushAlert record
    """
    if data is None:
        data = {}

    alert_id = hashlib.sha256(
        f"{device_id}:{title}:{int(time.time() * 1000)}".encode()
    ).hexdigest()[:16]

    alert = PushAlert(
        alert_id=alert_id,
        device_id=device_id,
        title=title,
        message=message,
        priority=priority,
        alert_type=alert_type,
        source=source,
        data=data,
    )

    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('''INSERT INTO alerts
                 (alert_id, device_id, title, message, priority, alert_type, source, created_at, data)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (alert_id, device_id, title, message, priority, alert_type, source,
               alert.created_at, json.dumps(data)))
    conn.commit()
    conn.close()

    return alert


def queue_alerts_from_proactive(device_id: str, alerts: List) -> int:
    """Queue alerts from proactive_intel.Alert objects.

    Args:
        device_id: Target device
        alerts: List of Alert objects from proactive_intel

    Returns:
        Number of alerts queued
    """
    queued = 0

    for alert in alerts:
        try:
            queue_alert_for_device(
                device_id=device_id,
                title=alert.title,
                message=alert.message,
                priority=alert.priority,
                alert_type=alert.alert_type,
                source=alert.source,
                data=alert.data,
            )
            queued += 1
        except Exception:
            pass

    return queued


# -- Alert Retrieval --

def get_pending_alerts(device_id: str, apply_quiet_hours: bool = True) -> List[PushAlert]:
    """Get pending alerts for a device, respecting preferences.

    Args:
        device_id: Device identifier
        apply_quiet_hours: Whether to filter by quiet hours

    Returns:
        List of pending alerts
    """
    prefs = get_preferences(device_id)

    # Check quiet hours
    if apply_quiet_hours and _in_quiet_hours(prefs):
        # Only return urgent alerts during quiet hours
        priority_filter = "urgent"
    else:
        priority_filter = prefs.min_priority

    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()

    # Priority order: urgent > high > medium > low
    priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
    min_order = priority_order.get(priority_filter, 2)

    priority_list = [p for p, o in priority_order.items() if o <= min_order]

    placeholders = ",".join("?" * len(priority_list))
    c.execute(f'''SELECT * FROM alerts
                  WHERE device_id = ? AND status = "pending" AND priority IN ({placeholders})
                  ORDER BY priority ASC, created_at DESC''',
              (device_id, *priority_list))
    rows = c.fetchall()
    conn.close()

    alerts = []
    for row in rows:
        try:
            data = json.loads(row[12]) if row[12] else {}
        except json.JSONDecodeError:
            data = {}

        alerts.append(PushAlert(
            alert_id=row[0],
            device_id=row[1],
            title=row[2],
            message=row[3],
            priority=row[4],
            alert_type=row[5],
            source=row[6],
            status=row[7],
            created_at=row[8],
            sent_at=row[9],
            delivered_at=row[10],
            read_at=row[11],
            data=data,
        ))

    return alerts


def mark_delivered(alert_id: str) -> bool:
    """Mark an alert as delivered."""
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('''UPDATE alerts SET status = "delivered", delivered_at = ?, sent_at = ?
                 WHERE alert_id = ?''', (now, now, alert_id))
    conn.commit()
    affected = c.rowcount
    conn.close()

    return affected > 0


def mark_read(alert_id: str) -> bool:
    """Mark an alert as read."""
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('''UPDATE alerts SET status = "read", read_at = ?
                 WHERE alert_id = ?''', (now, alert_id))
    conn.commit()
    affected = c.rowcount
    conn.close()

    return affected > 0


def dismiss_alert(alert_id: str) -> bool:
    """Dismiss an alert."""
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('''UPDATE alerts SET status = "dismissed"
                 WHERE alert_id = ?''', (alert_id,))
    conn.commit()
    affected = c.rowcount
    conn.close()

    return affected > 0


def get_alert_history(device_id: str, limit: int = 50) -> List[PushAlert]:
    """Get alert history for a device."""
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('''SELECT * FROM alerts WHERE device_id = ?
                 ORDER BY created_at DESC LIMIT ?''', (device_id, limit))
    rows = c.fetchall()
    conn.close()

    alerts = []
    for row in rows:
        try:
            data = json.loads(row[12]) if row[12] else {}
        except json.JSONDecodeError:
            data = {}

        alerts.append(PushAlert(
            alert_id=row[0],
            device_id=row[1],
            title=row[2],
            message=row[3],
            priority=row[4],
            alert_type=row[5],
            source=row[6],
            status=row[7],
            created_at=row[8],
            sent_at=row[9],
            delivered_at=row[10],
            read_at=row[11],
            data=data,
        ))

    return alerts


# -- Quiet Hours & Batch Delivery --

def _in_quiet_hours(prefs: AlertPreferences) -> bool:
    """Check if current time is in quiet hours."""
    if not prefs.quiet_hours_start or not prefs.quiet_hours_end:
        return False

    try:
        start = dt_time.fromisoformat(prefs.quiet_hours_start)
        end = dt_time.fromisoformat(prefs.quiet_hours_end)
        now = datetime.now().time()

        if start < end:
            return start <= now < end
        else:
            return now >= start or now < end
    except ValueError:
        return False


def apply_quiet_hours(device_id: str) -> int:
    """Apply quiet hour filtering to pending alerts.

    Returns:
        Number of alerts hidden
    """
    prefs = get_preferences(device_id)

    if not _in_quiet_hours(prefs):
        return 0

    # Move non-urgent alerts to a holding state
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('''UPDATE alerts SET status = "held"
                 WHERE device_id = ? AND status = "pending" AND priority != "urgent"''',
              (device_id,))
    conn.commit()
    affected = c.rowcount
    conn.close()

    return affected


def create_digest(device_id: str) -> Optional[dict]:
    """Create a low-priority alert digest for a device.

    Returns:
        Digest alert dict or None if no low-priority alerts
    """
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('''SELECT alert_id FROM alerts
                 WHERE device_id = ? AND status IN ("pending", "held") AND priority IN ("medium", "low")
                 ORDER BY created_at DESC LIMIT 20''', (device_id,))
    rows = c.fetchall()
    conn.close()

    if not rows:
        return None

    alert_ids = [row[0] for row in rows]
    digest_id = hashlib.sha256(
        f"{device_id}:{int(time.time())}".encode()
    ).hexdigest()[:16]

    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('''INSERT INTO digests (digest_id, device_id, alert_ids, sent_at)
                 VALUES (?, ?, ?, ?)''',
              (digest_id, device_id, json.dumps(alert_ids), now))
    conn.commit()
    conn.close()

    return {
        "digest_id": digest_id,
        "device_id": device_id,
        "alert_count": len(alert_ids),
        "sent_at": now,
    }


# -- Utility --

def get_queue_stats(device_id: str) -> dict:
    """Get alert queue statistics for a device."""
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()

    c.execute('SELECT status, COUNT(*) FROM alerts WHERE device_id = ? GROUP BY status',
              (device_id,))
    status_counts = {row[0]: row[1] for row in c.fetchall()}

    c.execute('SELECT priority, COUNT(*) FROM alerts WHERE device_id = ? AND status = ? GROUP BY priority',
              (device_id, "pending"))
    priority_counts = {row[0]: row[1] for row in c.fetchall()}

    conn.close()

    return {
        "by_status": status_counts,
        "pending_by_priority": priority_counts,
        "total_pending": status_counts.get("pending", 0),
    }


def clear_old_alerts(device_id: str, days_old: int = 30) -> int:
    """Clear alerts older than specified days.

    Args:
        device_id: Device to clear alerts for
        days_old: Age threshold in days

    Returns:
        Number of alerts deleted
    """
    cutoff = datetime.now(timezone.utc).isoformat()
    # Simple comparison on ISO strings (yyyy-mm-dd is sortable)
    cutoff_date = cutoff[:10]

    from datetime import timedelta
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()[:10]

    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('''DELETE FROM alerts WHERE device_id = ? AND created_at < ?''',
              (device_id, cutoff_date))
    conn.commit()
    affected = c.rowcount
    conn.close()

    return affected


# Module-level convenience wrapper functions
# Rename original functions and create new wrappers with expected signatures

_original_get_pending_alerts = get_pending_alerts
_original_queue_alert_for_device = queue_alert_for_device
_original_register_device = register_device
_original_mark_delivered = mark_delivered
_original_configure_preferences = configure_preferences


def get_pending_alerts(device_id, limit=50):
    """Get pending alerts for a device.

    Args:
        device_id: Device identifier
        limit: Max alerts to return (new parameter)

    Returns:
        list of alert dicts
    """
    alerts = _original_get_pending_alerts(device_id)
    return [a.to_dict() for a in alerts[:limit]]


def queue_alert_for_device(device_id, alert):
    """Queue an alert for a device.

    Args:
        device_id: Target device ID
        alert: Alert dict or object with title, message, priority

    Returns:
        dict with alert record
    """
    if isinstance(alert, dict):
        result = _original_queue_alert_for_device(
            device_id,
            alert.get('title', 'Alert'),
            alert.get('message', ''),
            alert.get('priority', 'medium'),
            alert.get('alert_type', 'general'),
            alert.get('source', ''),
            alert.get('data')
        )
    else:
        result = _original_queue_alert_for_device(
            device_id,
            getattr(alert, 'title', 'Alert'),
            getattr(alert, 'message', ''),
            getattr(alert, 'priority', 'medium'),
            getattr(alert, 'alert_type', 'general'),
            getattr(alert, 'source', '')
        )
    return result.to_dict()


def register_device(device_id, device_name="", platform="unknown"):
    """Register a mobile device.

    Args:
        device_id: Device identifier
        device_name: Human-readable device name
        platform: Device platform (ios, android, unknown)

    Returns:
        dict with device record
    """
    device = _original_register_device(device_id, device_name or device_id, platform)
    return device.to_dict()


def mark_delivered(alert_id):
    """Mark an alert as delivered.

    Args:
        alert_id: Alert identifier

    Returns:
        None
    """
    _original_mark_delivered(alert_id)


def configure_preferences(device_id, min_priority="medium", quiet_start=None, quiet_end=None):
    """Configure alert preferences for a device.

    Args:
        device_id: Device identifier
        min_priority: Minimum priority level
        quiet_start: Quiet hours start time (HH:MM)
        quiet_end: Quiet hours end time (HH:MM)

    Returns:
        dict with preferences
    """
    prefs = _original_configure_preferences(device_id, min_priority, quiet_start, quiet_end)
    return prefs.to_dict()

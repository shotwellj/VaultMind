"""
VaultMind Proactive Intelligence
Phase 4 -- VaultMind finds you when something matters. Push, not just pull.

Monitors:
  1. Watched folders for new/changed documents
  2. Scheduled web searches for topic changes
  3. Deadline tracking from indexed documents
  4. Document change detection (diff summaries)

When something important happens, it creates an alert that the frontend
can display as a notification or digest.

Storage: ~/.vaultmind/proactive/
No cloud. Everything local.
"""

import os
import re
import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict


# ── Config ────────────────────────────────────────────────────

PROACTIVE_DIR = os.path.expanduser("~/.vaultmind/proactive")
ALERTS_PATH = os.path.join(PROACTIVE_DIR, "alerts.json")
WATCHES_PATH = os.path.join(PROACTIVE_DIR, "watches.json")
FILE_HASHES_PATH = os.path.join(PROACTIVE_DIR, "file_hashes.json")
DEADLINES_PATH = os.path.join(PROACTIVE_DIR, "deadlines.json")

# File extensions to monitor
WATCHABLE_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".csv",
    ".xlsx", ".xls", ".pptx", ".json", ".html",
}


# ── Data Classes ──────────────────────────────────────────────

class AlertPriority:
    URGENT = "urgent"  # Deadline approaching, critical change
    HIGH = "high"  # New document in watched folder
    MEDIUM = "medium"  # Topic update from web
    LOW = "low"  # Minor file change


class AlertType:
    NEW_DOCUMENT = "new_document"
    DOCUMENT_CHANGED = "document_changed"
    DEADLINE_APPROACHING = "deadline_approaching"
    TOPIC_UPDATE = "topic_update"
    FOLDER_CHANGE = "folder_change"


@dataclass
class Alert:
    id: str
    alert_type: str
    priority: str
    title: str
    message: str
    source: str = ""  # file path, URL, or topic
    timestamp: str = ""
    read: bool = False
    dismissed: bool = False
    data: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if not self.id:
            self.id = hashlib.md5(
                f"{self.alert_type}:{self.source}:{self.timestamp}".encode()
            ).hexdigest()[:12]


@dataclass
class FolderWatch:
    path: str
    label: str = ""
    active: bool = True
    created: str = ""
    last_scanned: str = ""

    def __post_init__(self):
        if not self.created:
            self.created = datetime.utcnow().isoformat()
        if not self.label:
            self.label = os.path.basename(self.path)


@dataclass
class TopicWatch:
    topic: str
    search_query: str = ""
    interval_hours: int = 24
    active: bool = True
    created: str = ""
    last_checked: str = ""

    def __post_init__(self):
        if not self.created:
            self.created = datetime.utcnow().isoformat()
        if not self.search_query:
            self.search_query = self.topic


@dataclass
class Deadline:
    title: str
    date: str  # ISO date string
    source: str = ""  # Document it was extracted from
    alert_days: list = field(default_factory=lambda: [7, 3, 1])  # Days before to alert
    acknowledged: bool = False


# ── Persistence ───────────────────────────────────────────────

def _ensure_dir():
    os.makedirs(PROACTIVE_DIR, exist_ok=True)


def _load_json(path: str, default=None):
    if default is None:
        default = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _save_json(path: str, data):
    _ensure_dir()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Alert Management ─────────────────────────────────────────

def get_alerts(include_dismissed: bool = False, limit: int = 50) -> list:
    """Get active alerts, newest first."""
    alerts = _load_json(ALERTS_PATH, [])
    if not include_dismissed:
        alerts = [a for a in alerts if not a.get("dismissed")]
    alerts.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
    return alerts[:limit]


def get_unread_count() -> int:
    """Get the count of unread alerts."""
    alerts = _load_json(ALERTS_PATH, [])
    return sum(1 for a in alerts if not a.get("read") and not a.get("dismissed"))


def _add_alert(alert: Alert):
    """Add a new alert (deduplicates by source + type within 24 hours)."""
    alerts = _load_json(ALERTS_PATH, [])

    # Deduplicate: don't re-alert for the same source + type within 24 hours
    cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    for existing in alerts:
        if (existing.get("source") == alert.source
                and existing.get("alert_type") == alert.alert_type
                and existing.get("timestamp", "") > cutoff):
            return  # Already alerted recently

    alerts.append(asdict(alert))

    # Keep only the last 200 alerts
    if len(alerts) > 200:
        alerts = alerts[-200:]

    _save_json(ALERTS_PATH, alerts)


def mark_read(alert_id: str):
    """Mark an alert as read."""
    alerts = _load_json(ALERTS_PATH, [])
    for a in alerts:
        if a.get("id") == alert_id:
            a["read"] = True
    _save_json(ALERTS_PATH, alerts)


def dismiss_alert(alert_id: str):
    """Dismiss an alert."""
    alerts = _load_json(ALERTS_PATH, [])
    for a in alerts:
        if a.get("id") == alert_id:
            a["dismissed"] = True
    _save_json(ALERTS_PATH, alerts)


def mark_all_read():
    """Mark all alerts as read."""
    alerts = _load_json(ALERTS_PATH, [])
    for a in alerts:
        a["read"] = True
    _save_json(ALERTS_PATH, alerts)


# ── Folder Watching ───────────────────────────────────────────

def add_folder_watch(path: str, label: str = "") -> dict:
    """Add a folder to the watch list."""
    watches = _load_json(WATCHES_PATH, {"folders": [], "topics": []})
    folders = watches.get("folders", [])

    # Check if already watching
    for f in folders:
        if f.get("path") == path:
            return {"status": "already_watching", "path": path}

    if not os.path.isdir(path):
        return {"error": f"Path does not exist: {path}"}

    watch = FolderWatch(path=path, label=label)
    folders.append(asdict(watch))
    watches["folders"] = folders
    _save_json(WATCHES_PATH, watches)

    # Do initial scan to establish baseline
    _scan_folder(path)

    return {"status": "watching", "path": path, "label": watch.label}


def remove_folder_watch(path: str) -> dict:
    """Remove a folder from the watch list."""
    watches = _load_json(WATCHES_PATH, {"folders": [], "topics": []})
    folders = watches.get("folders", [])
    watches["folders"] = [f for f in folders if f.get("path") != path]
    _save_json(WATCHES_PATH, watches)
    return {"status": "removed", "path": path}


def get_watches() -> dict:
    """Get all active watches."""
    return _load_json(WATCHES_PATH, {"folders": [], "topics": []})


def _file_hash(filepath: str) -> str:
    """Quick hash of file modification time + size (fast, no full read)."""
    try:
        stat = os.stat(filepath)
        return hashlib.md5(f"{stat.st_mtime}:{stat.st_size}".encode()).hexdigest()
    except Exception:
        return ""


def _scan_folder(folder_path: str) -> list:
    """Scan a folder and return list of changes since last scan.

    Returns list of change dicts: [{"type": "new"|"changed", "path": "...", "filename": "..."}]
    """
    hashes = _load_json(FILE_HASHES_PATH, {})
    changes = []

    if not os.path.isdir(folder_path):
        return changes

    current_files = {}
    for root, dirs, files in os.walk(folder_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in WATCHABLE_EXTENSIONS:
                continue
            filepath = os.path.join(root, filename)
            fhash = _file_hash(filepath)
            current_files[filepath] = fhash

            old_hash = hashes.get(filepath)
            if old_hash is None:
                # New file
                changes.append({
                    "type": "new",
                    "path": filepath,
                    "filename": filename,
                })
            elif old_hash != fhash:
                # Changed file
                changes.append({
                    "type": "changed",
                    "path": filepath,
                    "filename": filename,
                })

    # Update stored hashes
    for filepath, fhash in current_files.items():
        hashes[filepath] = fhash
    _save_json(FILE_HASHES_PATH, hashes)

    return changes


def scan_all_watches() -> list:
    """Scan all watched folders and generate alerts for changes.

    Returns list of new alerts generated.
    """
    watches = _load_json(WATCHES_PATH, {"folders": [], "topics": []})
    new_alerts = []

    for folder in watches.get("folders", []):
        if not folder.get("active", True):
            continue

        path = folder.get("path", "")
        label = folder.get("label", os.path.basename(path))

        changes = _scan_folder(path)
        for change in changes:
            filename = change["filename"]
            change_type = change["type"]

            if change_type == "new":
                alert = Alert(
                    id="",
                    alert_type=AlertType.NEW_DOCUMENT,
                    priority=AlertPriority.HIGH,
                    title=f"New document: {filename}",
                    message=f"A new file was added to {label}. It may contain information relevant to your work.",
                    source=change["path"],
                    data={"folder": path, "filename": filename},
                )
            else:
                alert = Alert(
                    id="",
                    alert_type=AlertType.DOCUMENT_CHANGED,
                    priority=AlertPriority.MEDIUM,
                    title=f"Document updated: {filename}",
                    message=f"{filename} in {label} was modified. Review for new or changed information.",
                    source=change["path"],
                    data={"folder": path, "filename": filename},
                )

            _add_alert(alert)
            new_alerts.append(asdict(alert))

        # Update last scanned time
        folder["last_scanned"] = datetime.utcnow().isoformat()

    watches["folders"] = watches.get("folders", [])
    _save_json(WATCHES_PATH, watches)

    return new_alerts


# ── Topic Watching ────────────────────────────────────────────

def add_topic_watch(topic: str, search_query: str = "", interval_hours: int = 24) -> dict:
    """Add a topic to monitor via scheduled web searches."""
    watches = _load_json(WATCHES_PATH, {"folders": [], "topics": []})
    topics = watches.get("topics", [])

    # Check if already watching
    for t in topics:
        if t.get("topic", "").lower() == topic.lower():
            return {"status": "already_watching", "topic": topic}

    watch = TopicWatch(topic=topic, search_query=search_query, interval_hours=interval_hours)
    topics.append(asdict(watch))
    watches["topics"] = topics
    _save_json(WATCHES_PATH, watches)

    return {"status": "watching", "topic": topic, "interval": f"every {interval_hours}h"}


def remove_topic_watch(topic: str) -> dict:
    """Remove a topic from the watch list."""
    watches = _load_json(WATCHES_PATH, {"folders": [], "topics": []})
    topics = watches.get("topics", [])
    watches["topics"] = [t for t in topics if t.get("topic", "").lower() != topic.lower()]
    _save_json(WATCHES_PATH, watches)
    return {"status": "removed", "topic": topic}


# ── Deadline Tracking ─────────────────────────────────────────

DATE_PATTERNS = [
    re.compile(r'\b(?:deadline|due|expires?|by)\s*:?\s*(\w+\s+\d{1,2},?\s+\d{4})', re.IGNORECASE),
    re.compile(r'\b(?:deadline|due|expires?|by)\s*:?\s*(\d{4}-\d{2}-\d{2})', re.IGNORECASE),
    re.compile(r'\b(?:must\s+(?:be\s+)?(?:completed?|submitted?|filed?)\s+(?:by|before))\s+(\w+\s+\d{1,2},?\s+\d{4})', re.IGNORECASE),
]


def extract_deadlines(text: str, source: str = "") -> list:
    """Extract deadlines from document text.

    Returns a list of Deadline dicts.
    """
    deadlines = []
    seen = set()

    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            date_str = match.group(1).strip()
            if date_str in seen:
                continue
            seen.add(date_str)

            # Try to parse the date
            parsed = _parse_date(date_str)
            if parsed:
                # Get surrounding context for the title
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip().replace("\n", " ")

                deadlines.append(asdict(Deadline(
                    title=context[:100],
                    date=parsed.isoformat(),
                    source=source,
                )))

    return deadlines


def _parse_date(date_str: str) -> Optional[datetime]:
    """Try to parse a date string."""
    formats = [
        "%B %d, %Y",   # January 15, 2024
        "%B %d %Y",    # January 15 2024
        "%b %d, %Y",   # Jan 15, 2024
        "%Y-%m-%d",    # 2024-01-15
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def add_deadline(title: str, date: str, source: str = "", alert_days: list = None) -> dict:
    """Manually add a deadline to track."""
    if alert_days is None:
        alert_days = [7, 3, 1]

    deadlines = _load_json(DEADLINES_PATH, [])
    deadline = Deadline(title=title, date=date, source=source, alert_days=alert_days)
    deadlines.append(asdict(deadline))
    _save_json(DEADLINES_PATH, deadlines)
    return {"status": "added", "title": title, "date": date}


def check_deadlines() -> list:
    """Check all tracked deadlines and generate alerts for approaching ones.

    Returns list of new alerts.
    """
    deadlines = _load_json(DEADLINES_PATH, [])
    now = datetime.utcnow()
    new_alerts = []

    for dl in deadlines:
        if dl.get("acknowledged"):
            continue

        try:
            deadline_date = datetime.fromisoformat(dl["date"])
        except (ValueError, KeyError):
            continue

        days_until = (deadline_date - now).days

        # Check if we should alert
        alert_days = dl.get("alert_days", [7, 3, 1])
        for threshold in alert_days:
            if days_until == threshold or (days_until < 0 and threshold == 1):
                if days_until < 0:
                    priority = AlertPriority.URGENT
                    title = f"OVERDUE: {dl['title']}"
                    message = f"This deadline was {abs(days_until)} days ago!"
                elif days_until <= 1:
                    priority = AlertPriority.URGENT
                    title = f"Tomorrow: {dl['title']}"
                    message = f"Deadline is tomorrow ({dl['date']})"
                elif days_until <= 3:
                    priority = AlertPriority.HIGH
                    title = f"{days_until} days: {dl['title']}"
                    message = f"Deadline in {days_until} days ({dl['date']})"
                else:
                    priority = AlertPriority.MEDIUM
                    title = f"{days_until} days: {dl['title']}"
                    message = f"Deadline approaching: {dl['date']}"

                alert = Alert(
                    id="",
                    alert_type=AlertType.DEADLINE_APPROACHING,
                    priority=priority,
                    title=title[:80],
                    message=message,
                    source=dl.get("source", ""),
                    data={"deadline": dl},
                )
                _add_alert(alert)
                new_alerts.append(asdict(alert))
                break  # Only one alert per deadline per check

    return new_alerts


# ── Run All Checks ────────────────────────────────────────────

def run_proactive_scan() -> dict:
    """Run all proactive intelligence checks.

    Called periodically (e.g., every 5 minutes by a background thread,
    or on app startup).

    Returns summary of findings.
    """
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "folder_alerts": [],
        "deadline_alerts": [],
        "total_new_alerts": 0,
    }

    # Scan watched folders
    try:
        folder_alerts = scan_all_watches()
        results["folder_alerts"] = folder_alerts
        results["total_new_alerts"] += len(folder_alerts)
    except Exception as e:
        results["folder_error"] = str(e)

    # Check deadlines
    try:
        deadline_alerts = check_deadlines()
        results["deadline_alerts"] = deadline_alerts
        results["total_new_alerts"] += len(deadline_alerts)
    except Exception as e:
        results["deadline_error"] = str(e)

    return results


def get_proactive_summary() -> dict:
    """Get a summary of proactive intelligence status.

    Useful for the daily war room / morning briefing.
    """
    watches = _load_json(WATCHES_PATH, {"folders": [], "topics": []})
    deadlines = _load_json(DEADLINES_PATH, [])
    unread = get_unread_count()

    # Upcoming deadlines (next 30 days)
    now = datetime.utcnow()
    upcoming = []
    for dl in deadlines:
        if dl.get("acknowledged"):
            continue
        try:
            deadline_date = datetime.fromisoformat(dl["date"])
            days = (deadline_date - now).days
            if 0 <= days <= 30:
                upcoming.append({
                    "title": dl["title"][:60],
                    "date": dl["date"],
                    "days_until": days,
                    "source": dl.get("source", ""),
                })
        except (ValueError, KeyError):
            continue

    upcoming.sort(key=lambda x: x["days_until"])

    return {
        "unread_alerts": unread,
        "watched_folders": len([f for f in watches.get("folders", []) if f.get("active")]),
        "watched_topics": len([t for t in watches.get("topics", []) if t.get("active")]),
        "tracked_deadlines": len([d for d in deadlines if not d.get("acknowledged")]),
        "upcoming_deadlines": upcoming[:5],
    }

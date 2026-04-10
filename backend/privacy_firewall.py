"""
VaultMind Privacy Firewall
Phase 2 -- strips private entities before anything touches the network.

This is NOT a policy. It's a wall. NER-based entity detection catches:
  - Person names
  - Email addresses
  - Phone numbers
  - Dollar amounts / financial figures
  - Case/matter numbers
  - Social Security Numbers
  - Credit card numbers
  - Dates (configurable)
  - Internal document titles / file paths
  - Custom blocklist terms (client names, project codenames, etc.)

Architecture:
  User query --> Privacy Firewall --> Sanitized query --> Web search
  Web results --> Context Fusion --> Full context reapplied LOCALLY

Every strip decision is logged to an audit trail for compliance review.

CROSS-PRODUCT NOTE:
  This module is designed to also work as an AIR Blackbox component.
  It maps to EU AI Act Article 10 (Data Governance) -- preventing
  private/sensitive data from leaking through AI pipelines.
  The same engine can scan prompts, tool calls, and agent outputs.
"""

import re
import os
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


# ── Entity Types ───────────────────────────────────────────────

class EntityType(str, Enum):
    PERSON = "person"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    MONEY = "money"
    DATE = "date"
    CASE_NUMBER = "case_number"
    FILE_PATH = "file_path"
    IP_ADDRESS = "ip_address"
    CUSTOM = "custom"  # From user blocklist


@dataclass
class DetectedEntity:
    """A private entity found in the text."""
    entity_type: EntityType
    value: str
    start: int
    end: int
    replacement: str  # What it was replaced with
    confidence: float = 1.0


@dataclass
class FirewallResult:
    """Result of running the privacy firewall on text."""
    original_text: str
    sanitized_text: str
    entities_found: list = field(default_factory=list)
    entity_count: int = 0
    was_modified: bool = False
    audit_id: str = ""
    timestamp: str = ""

    def to_dict(self):
        d = asdict(self)
        # Convert enum values to strings for JSON
        for e in d["entities_found"]:
            if isinstance(e["entity_type"], EntityType):
                e["entity_type"] = e["entity_type"].value
        return d


# ── Configuration ──────────────────────────────────────────────

FIREWALL_DIR = os.environ.get(
    "VAULTMIND_FIREWALL_DIR",
    os.path.expanduser("~/.vaultmind/firewall")
)
BLOCKLIST_FILE = os.path.join(FIREWALL_DIR, "blocklist.json")
AUDIT_LOG_FILE = os.path.join(FIREWALL_DIR, "audit_log.jsonl")
CONFIG_FILE = os.path.join(FIREWALL_DIR, "config.json")

os.makedirs(FIREWALL_DIR, exist_ok=True)

# Default config
DEFAULT_CONFIG = {
    "enabled": True,
    "strip_persons": True,
    "strip_emails": True,
    "strip_phones": True,
    "strip_ssn": True,
    "strip_credit_cards": True,
    "strip_money": True,
    "strip_dates": False,  # Off by default -- too aggressive for most queries
    "strip_case_numbers": True,
    "strip_file_paths": True,
    "strip_ip_addresses": True,
    "strip_custom_blocklist": True,
    "audit_logging": True,
    "replacement_style": "bracket",  # "bracket" = [PERSON], "redact" = ████
}


# ── Regex Patterns ─────────────────────────────────────────────
# These run BEFORE spaCy NER as a fast first pass.
# spaCy catches what regex misses (contextual names, etc.)

PATTERNS = {
    EntityType.EMAIL: re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ),
    EntityType.PHONE: re.compile(
        r'(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    ),
    EntityType.SSN: re.compile(
        r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
    ),
    EntityType.CREDIT_CARD: re.compile(
        r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
    ),
    EntityType.MONEY: re.compile(
        r'\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b'
        r'|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s?(?:dollars?|USD|EUR|GBP)\b',
        re.IGNORECASE
    ),
    EntityType.CASE_NUMBER: re.compile(
        r'\b(?:Case|Matter|Docket|File)\s*(?:#|No\.?|Number)?\s*[:.]?\s*'
        r'[A-Z0-9][-A-Z0-9./:]{3,20}\b',
        re.IGNORECASE
    ),
    EntityType.FILE_PATH: re.compile(
        r'(?:/[A-Za-z0-9._-]+){2,}'  # Unix paths
        r'|[A-Z]:\\(?:[A-Za-z0-9._-]+\\){1,}'  # Windows paths
        r'|\b\w+\.(?:pdf|docx?|xlsx?|pptx?|txt|csv|json)\b',  # File names
        re.IGNORECASE
    ),
    EntityType.IP_ADDRESS: re.compile(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ),
    EntityType.DATE: re.compile(
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
        r'\s+\d{1,2},?\s+\d{4}\b'
        r'|\b\d{1,2}/\d{1,2}/\d{2,4}\b'
        r'|\b\d{4}-\d{2}-\d{2}\b',
        re.IGNORECASE
    ),
}

# Common person name patterns (supplements spaCy NER)
# Catches "John Smith", "J. Smith", "Dr. Smith", etc.
PERSON_PATTERN = re.compile(
    r'\b(?:Mr|Mrs|Ms|Dr|Prof|Judge|Hon|Atty)\.?\s+'
    r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b'
)


# ── spaCy NER (optional, loaded lazily) ────────────────────────

_nlp = None
_spacy_available = None


def _get_nlp():
    """Lazy-load spaCy with the English NER model."""
    global _nlp, _spacy_available
    if _spacy_available is not None:
        return _nlp

    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
            _spacy_available = True
            print("[PrivacyFirewall] spaCy NER loaded (en_core_web_sm)")
        except OSError:
            # Model not installed
            print("[PrivacyFirewall] spaCy model not found. Run: python -m spacy download en_core_web_sm")
            print("[PrivacyFirewall] Falling back to regex-only mode (still catches most entities)")
            _spacy_available = False
    except ImportError:
        print("[PrivacyFirewall] spaCy not installed. Run: pip install spacy")
        print("[PrivacyFirewall] Falling back to regex-only mode")
        _spacy_available = False

    return _nlp


# ── Core Firewall Logic ───────────────────────────────────────

def load_config() -> dict:
    """Load firewall configuration."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
                return {**DEFAULT_CONFIG, **cfg}
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    """Save firewall configuration."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def load_blocklist() -> list[str]:
    """Load the custom blocklist (client names, project codenames, etc.)."""
    if os.path.exists(BLOCKLIST_FILE):
        try:
            with open(BLOCKLIST_FILE) as f:
                data = json.load(f)
                return data.get("terms", [])
        except (json.JSONDecodeError, IOError):
            pass
    return []


def save_blocklist(terms: list[str]):
    """Save the custom blocklist."""
    with open(BLOCKLIST_FILE, "w") as f:
        json.dump({"terms": terms, "updated_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)


def sanitize(text: str, config: Optional[dict] = None) -> FirewallResult:
    """Run the privacy firewall on text.

    This is the main entry point. Call this before any web search.

    Args:
        text: The raw user query or text to sanitize
        config: Optional config override (uses saved config if None)

    Returns:
        FirewallResult with sanitized text, detected entities, and audit info
    """
    if config is None:
        config = load_config()

    if not config.get("enabled", True):
        return FirewallResult(
            original_text=text,
            sanitized_text=text,
            entity_count=0,
            was_modified=False,
        )

    entities = []
    replacement_style = config.get("replacement_style", "bracket")

    # ── Pass 1: Regex-based detection (fast) ──────────────────

    type_to_config_key = {
        EntityType.EMAIL: "strip_emails",
        EntityType.PHONE: "strip_phones",
        EntityType.SSN: "strip_ssn",
        EntityType.CREDIT_CARD: "strip_credit_cards",
        EntityType.MONEY: "strip_money",
        EntityType.DATE: "strip_dates",
        EntityType.CASE_NUMBER: "strip_case_numbers",
        EntityType.FILE_PATH: "strip_file_paths",
        EntityType.IP_ADDRESS: "strip_ip_addresses",
    }

    for entity_type, pattern in PATTERNS.items():
        config_key = type_to_config_key.get(entity_type)
        if config_key and not config.get(config_key, True):
            continue

        for match in pattern.finditer(text):
            replacement = _make_replacement(entity_type, replacement_style)
            entities.append(DetectedEntity(
                entity_type=entity_type,
                value=match.group(),
                start=match.start(),
                end=match.end(),
                replacement=replacement,
                confidence=0.95,
            ))

    # ── Pass 1.5: Titled person names via regex ───────────────

    if config.get("strip_persons", True):
        for match in PERSON_PATTERN.finditer(text):
            replacement = _make_replacement(EntityType.PERSON, replacement_style)
            entities.append(DetectedEntity(
                entity_type=EntityType.PERSON,
                value=match.group(),
                start=match.start(),
                end=match.end(),
                replacement=replacement,
                confidence=0.85,
            ))

    # ── Pass 2: spaCy NER (catches contextual names) ─────────

    if config.get("strip_persons", True):
        nlp = _get_nlp()
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Check not already caught by regex
                    already_caught = any(
                        e.start <= ent.start_char and e.end >= ent.end_char
                        for e in entities
                    )
                    if not already_caught:
                        replacement = _make_replacement(EntityType.PERSON, replacement_style)
                        entities.append(DetectedEntity(
                            entity_type=EntityType.PERSON,
                            value=ent.text,
                            start=ent.start_char,
                            end=ent.end_char,
                            replacement=replacement,
                            confidence=0.80,
                        ))

    # ── Pass 3: Custom blocklist ──────────────────────────────

    if config.get("strip_custom_blocklist", True):
        blocklist = load_blocklist()
        for term in blocklist:
            if not term.strip():
                continue
            # Case-insensitive search
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            for match in pattern.finditer(text):
                already_caught = any(
                    e.start <= match.start() and e.end >= match.end()
                    for e in entities
                )
                if not already_caught:
                    replacement = _make_replacement(EntityType.CUSTOM, replacement_style)
                    entities.append(DetectedEntity(
                        entity_type=EntityType.CUSTOM,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        replacement=replacement,
                        confidence=1.0,
                    ))

    # ── Apply replacements (reverse order to preserve positions) ──

    entities.sort(key=lambda e: e.start, reverse=True)

    # Remove overlapping entities (keep highest confidence)
    entities = _deduplicate_entities(entities)

    sanitized = text
    for entity in entities:
        sanitized = sanitized[:entity.start] + entity.replacement + sanitized[entity.end:]

    # ── Build result ──────────────────────────────────────────

    audit_id = hashlib.sha256(
        f"{text}-{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()[:12]

    result = FirewallResult(
        original_text=text,
        sanitized_text=sanitized,
        entities_found=entities,
        entity_count=len(entities),
        was_modified=len(entities) > 0,
        audit_id=audit_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # ── Audit log ─────────────────────────────────────────────

    if config.get("audit_logging", True) and entities:
        _write_audit_log(result)

    return result


def sanitize_for_search(query: str, config: Optional[dict] = None) -> tuple[str, FirewallResult]:
    """Convenience wrapper: sanitize a query and return both the clean query and full result.

    Use this before passing a query to any web search function.

    Returns:
        (sanitized_query, firewall_result)
    """
    result = sanitize(query, config)
    return result.sanitized_text, result


# ── AIR Blackbox Integration ──────────────────────────────────
# These functions make the firewall usable as an AIR Blackbox component
# for Article 10 (Data Governance) compliance scanning.

def scan_for_data_leakage(text: str, context: str = "prompt") -> dict:
    """Scan text for potential private data leakage.

    This is the AIR Blackbox-compatible interface. It returns findings
    in the same format as other AIR Blackbox scanners.

    Args:
        text: Text to scan (could be a prompt, tool call, or agent output)
        context: Where this text appears ("prompt", "tool_call", "output", "log")

    Returns:
        AIR Blackbox-compatible finding dict:
        {
            "article": "10",
            "check": "data_governance_leakage",
            "severity": "HIGH" | "MEDIUM" | "LOW",
            "passed": bool,
            "entities_found": [...],
            "recommendation": str,
        }
    """
    # Run firewall in detection-only mode (don't strip, just find)
    config = load_config()
    config["enabled"] = True
    result = sanitize(text, config)

    # Determine severity based on what was found
    entity_types = set(e.entity_type for e in result.entities_found)

    if EntityType.SSN in entity_types or EntityType.CREDIT_CARD in entity_types:
        severity = "CRITICAL"
    elif EntityType.PERSON in entity_types or EntityType.EMAIL in entity_types:
        severity = "HIGH"
    elif EntityType.MONEY in entity_types or EntityType.PHONE in entity_types:
        severity = "MEDIUM"
    elif result.entity_count > 0:
        severity = "LOW"
    else:
        severity = "PASS"

    return {
        "article": "10",
        "article_title": "Data Governance",
        "check": "data_governance_leakage",
        "context": context,
        "severity": severity,
        "passed": result.entity_count == 0,
        "entity_count": result.entity_count,
        "entities_found": [
            {
                "type": e.entity_type.value if isinstance(e.entity_type, EntityType) else e.entity_type,
                "value_hash": hashlib.sha256(e.value.encode()).hexdigest()[:8],  # Hash, don't log the actual value
                "confidence": e.confidence,
            }
            for e in result.entities_found
        ],
        "recommendation": _build_recommendation(result, context),
        "evidence": f"Found {result.entity_count} potential PII entities in {context}",
    }


# ── Helper Functions ──────────────────────────────────────────

def _make_replacement(entity_type: EntityType, style: str = "bracket") -> str:
    """Generate a replacement string for a detected entity."""
    labels = {
        EntityType.PERSON: "PERSON",
        EntityType.EMAIL: "EMAIL",
        EntityType.PHONE: "PHONE",
        EntityType.SSN: "SSN",
        EntityType.CREDIT_CARD: "CARD",
        EntityType.MONEY: "AMOUNT",
        EntityType.DATE: "DATE",
        EntityType.CASE_NUMBER: "CASE_ID",
        EntityType.FILE_PATH: "FILE",
        EntityType.IP_ADDRESS: "IP",
        EntityType.CUSTOM: "REDACTED",
    }
    label = labels.get(entity_type, "REDACTED")

    if style == "redact":
        return "\u2588" * 6  # Block characters
    return f"[{label}]"


def _deduplicate_entities(entities: list[DetectedEntity]) -> list[DetectedEntity]:
    """Remove overlapping entity detections, keeping the highest confidence one."""
    if not entities:
        return entities

    # Sort by start position (already reverse-sorted, so re-sort forward)
    sorted_ents = sorted(entities, key=lambda e: (e.start, -e.confidence))

    result = [sorted_ents[0]]
    for ent in sorted_ents[1:]:
        prev = result[-1]
        # If overlapping, keep the one with higher confidence
        if ent.start < prev.end:
            if ent.confidence > prev.confidence:
                result[-1] = ent
        else:
            result.append(ent)

    # Re-sort in reverse order for replacement
    result.sort(key=lambda e: e.start, reverse=True)
    return result


def _write_audit_log(result: FirewallResult):
    """Append a strip decision to the audit log (JSONL format)."""
    entry = {
        "audit_id": result.audit_id,
        "timestamp": result.timestamp,
        "entity_count": result.entity_count,
        "entity_types": list(set(
            e.entity_type.value if isinstance(e.entity_type, EntityType) else e.entity_type
            for e in result.entities_found
        )),
        # Log hashes of stripped values, NOT the values themselves
        "stripped_hashes": [
            hashlib.sha256(e.value.encode()).hexdigest()[:8]
            for e in result.entities_found
        ],
        "original_length": len(result.original_text),
        "sanitized_length": len(result.sanitized_text),
    }
    try:
        with open(AUDIT_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except IOError as e:
        print(f"[PrivacyFirewall] Audit log write failed: {e}")


def _build_recommendation(result: FirewallResult, context: str) -> str:
    """Build a human-readable recommendation for AIR Blackbox reports."""
    if result.entity_count == 0:
        return f"No PII detected in {context}. Data governance check passed."

    types = set(
        e.entity_type.value if isinstance(e.entity_type, EntityType) else e.entity_type
        for e in result.entities_found
    )
    type_str = ", ".join(sorted(types))

    return (
        f"Detected {result.entity_count} potential PII entities ({type_str}) in {context}. "
        f"Implement a privacy firewall to strip sensitive data before it reaches external services. "
        f"Use entity detection (regex + NER) with a configurable blocklist and audit logging."
    )


def get_audit_log(limit: int = 50) -> list[dict]:
    """Read recent audit log entries."""
    if not os.path.exists(AUDIT_LOG_FILE):
        return []
    entries = []
    try:
        with open(AUDIT_LOG_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except IOError:
        pass
    return entries[-limit:]

"""
Contact Intelligence Briefings for VaultMind

Map contacts to VaultMind knowledge and generate pre-call briefings:
  - Contact storage: name, phone, email, company, role, tags, workspaces
  - Import from JSON (phone contact export format)
  - Contact-to-workspace mapping
  - Pre-call briefing generation
  - Contact search by name, company, phone, email
  - Interaction logging
  - Relationship strength scoring
  - Storage: SQLite at ~/.vaultmind/contacts/

Storage: ~/.vaultmind/contacts/
"""

import os
import json
import sqlite3
import hashlib
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from enum import Enum

CONTACTS_DIR = os.environ.get("VAULTMIND_CONTACTS_DIR", os.path.expanduser("~/.vaultmind/contacts"))
CONTACTS_DB = os.path.join(CONTACTS_DIR, "contacts.db")

os.makedirs(CONTACTS_DIR, exist_ok=True)


class InteractionType(str, Enum):
    """Types of contact interactions."""
    CALL = "call"
    EMAIL = "email"
    MEETING = "meeting"
    MESSAGE = "message"
    NOTE = "note"
    DOCUMENT_SHARED = "document_shared"


@dataclass
class Contact:
    """Contact record."""
    contact_id: str
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    tags: Optional[List[str]] = None
    workspaces: Optional[List[str]] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self):
        if not self.contact_id:
            self.contact_id = hashlib.sha256(
                f"{self.name}:{self.email or self.phone}:{int(time.time())}".encode()
            ).hexdigest()[:16]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        if self.tags is None:
            self.tags = []
        if self.workspaces is None:
            self.workspaces = []

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Interaction:
    """Record of a contact interaction."""
    interaction_id: str
    contact_id: str
    interaction_type: str
    summary: str
    date: Optional[str] = None
    duration_seconds: Optional[int] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if not self.interaction_id:
            self.interaction_id = hashlib.sha256(
                f"{self.contact_id}:{self.date}:{int(time.time() * 1000)}".encode()
            ).hexdigest()[:16]
        if not self.date:
            self.date = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Briefing:
    """Pre-call briefing for a contact."""
    contact_id: str
    contact_name: str
    company: Optional[str]
    role: Optional[str]
    last_interaction: Optional[str]
    open_action_items: List[str]
    relevant_documents: List[str]
    relationship_strength: float
    interaction_history: List[Dict]
    created_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# -- Database Initialization --

def _init_db():
    """Initialize contact database schema."""
    conn = sqlite3.connect(CONTACTS_DB)
    c = conn.cursor()

    # Contacts table
    c.execute('''CREATE TABLE IF NOT EXISTS contacts (
        contact_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        phone TEXT,
        email TEXT,
        company TEXT,
        role TEXT,
        tags TEXT,
        workspaces TEXT,
        notes TEXT,
        created_at TEXT,
        updated_at TEXT
    )''')

    # Interactions table
    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
        interaction_id TEXT PRIMARY KEY,
        contact_id TEXT NOT NULL,
        interaction_type TEXT,
        summary TEXT,
        date TEXT,
        duration_seconds INTEGER,
        created_at TEXT,
        FOREIGN KEY(contact_id) REFERENCES contacts(contact_id)
    )''')

    # Indexes for performance
    c.execute('CREATE INDEX IF NOT EXISTS idx_contacts_email ON contacts(email)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_contacts_phone ON contacts(phone)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_contacts_company ON contacts(company)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_interactions_contact ON interactions(contact_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_interactions_date ON interactions(date)')

    conn.commit()
    conn.close()


_init_db()


# -- Contact Management --

def add_contact(name: str, phone: Optional[str] = None, email: Optional[str] = None,
                company: Optional[str] = None, role: Optional[str] = None,
                tags: Optional[List[str]] = None, workspaces: Optional[List[str]] = None,
                notes: Optional[str] = None) -> Contact:
    """Add a new contact.

    Args:
        name: Contact name
        phone: Phone number
        email: Email address
        company: Company name
        role: Job role/title
        tags: List of tags
        workspaces: Associated workspaces
        notes: Additional notes

    Returns:
        Contact record
    """
    contact = Contact(
        contact_id="",
        name=name,
        phone=phone,
        email=email,
        company=company,
        role=role,
        tags=tags or [],
        workspaces=workspaces or [],
        notes=notes,
    )

    conn = sqlite3.connect(CONTACTS_DB)
    c = conn.cursor()
    c.execute('''INSERT INTO contacts
                 (contact_id, name, phone, email, company, role, tags, workspaces, notes, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (contact.contact_id, name, phone, email, company, role,
               json.dumps(contact.tags), json.dumps(contact.workspaces),
               notes, contact.created_at, contact.updated_at))
    conn.commit()
    conn.close()

    return contact


def get_contact(contact_id: str) -> Optional[Contact]:
    """Get a contact by ID."""
    conn = sqlite3.connect(CONTACTS_DB)
    c = conn.cursor()
    c.execute('SELECT * FROM contacts WHERE contact_id = ?', (contact_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return Contact(
            contact_id=row[0],
            name=row[1],
            phone=row[2],
            email=row[3],
            company=row[4],
            role=row[5],
            tags=json.loads(row[6]) if row[6] else [],
            workspaces=json.loads(row[7]) if row[7] else [],
            notes=row[8],
            created_at=row[9],
            updated_at=row[10],
        )
    return None


def search_contacts(query: str, field: Optional[str] = None) -> List[Contact]:
    """Search contacts by name, company, phone, or email.

    Args:
        query: Search query
        field: Specific field ("name", "company", "phone", "email") or None for all

    Returns:
        List of matching contacts
    """
    conn = sqlite3.connect(CONTACTS_DB)
    c = conn.cursor()

    query_lower = f"%{query.lower()}%"

    if field == "name":
        c.execute('SELECT * FROM contacts WHERE LOWER(name) LIKE ?', (query_lower,))
    elif field == "company":
        c.execute('SELECT * FROM contacts WHERE LOWER(company) LIKE ?', (query_lower,))
    elif field == "phone":
        c.execute('SELECT * FROM contacts WHERE LOWER(phone) LIKE ?', (query_lower,))
    elif field == "email":
        c.execute('SELECT * FROM contacts WHERE LOWER(email) LIKE ?', (query_lower,))
    else:
        # Search all fields
        c.execute('''SELECT * FROM contacts
                     WHERE LOWER(name) LIKE ? OR LOWER(company) LIKE ?
                     OR LOWER(phone) LIKE ? OR LOWER(email) LIKE ?''',
                  (query_lower, query_lower, query_lower, query_lower))

    rows = c.fetchall()
    conn.close()

    contacts = []
    for row in rows:
        contacts.append(Contact(
            contact_id=row[0],
            name=row[1],
            phone=row[2],
            email=row[3],
            company=row[4],
            role=row[5],
            tags=json.loads(row[6]) if row[6] else [],
            workspaces=json.loads(row[7]) if row[7] else [],
            notes=row[8],
            created_at=row[9],
            updated_at=row[10],
        ))

    return contacts


def update_contact(contact_id: str, **kwargs) -> Optional[Contact]:
    """Update contact fields."""
    contact = get_contact(contact_id)
    if not contact:
        return None

    for key, value in kwargs.items():
        if hasattr(contact, key):
            setattr(contact, key, value)

    contact.updated_at = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(CONTACTS_DB)
    c = conn.cursor()
    c.execute('''UPDATE contacts
                 SET name = ?, phone = ?, email = ?, company = ?, role = ?,
                     tags = ?, workspaces = ?, notes = ?, updated_at = ?
                 WHERE contact_id = ?''',
              (contact.name, contact.phone, contact.email, contact.company, contact.role,
               json.dumps(contact.tags), json.dumps(contact.workspaces),
               contact.notes, contact.updated_at, contact_id))
    conn.commit()
    conn.close()

    return contact


def list_contacts(limit: int = 100) -> List[Contact]:
    """List all contacts."""
    conn = sqlite3.connect(CONTACTS_DB)
    c = conn.cursor()
    c.execute('SELECT * FROM contacts ORDER BY updated_at DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()

    contacts = []
    for row in rows:
        contacts.append(Contact(
            contact_id=row[0],
            name=row[1],
            phone=row[2],
            email=row[3],
            company=row[4],
            role=row[5],
            tags=json.loads(row[6]) if row[6] else [],
            workspaces=json.loads(row[7]) if row[7] else [],
            notes=row[8],
            created_at=row[9],
            updated_at=row[10],
        ))

    return contacts


# -- Import Contacts --

def import_contacts(json_file_path: str) -> int:
    """Import contacts from JSON file (phone contact export format).

    Expects array of objects with fields:
      {name, phone, email, company, role, tags, notes}

    Args:
        json_file_path: Path to JSON file

    Returns:
        Number of contacts imported
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"File not found: {json_file_path}")

    with open(json_file_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    imported = 0
    for item in data:
        try:
            add_contact(
                name=item.get("name", ""),
                phone=item.get("phone"),
                email=item.get("email"),
                company=item.get("company"),
                role=item.get("role"),
                tags=item.get("tags", []),
                notes=item.get("notes"),
            )
            imported += 1
        except Exception:
            pass

    return imported


# -- Interactions --

def log_interaction(contact_id: str, interaction_type: str, summary: str,
                    date: Optional[str] = None, duration_seconds: Optional[int] = None) -> Interaction:
    """Log an interaction with a contact.

    Args:
        contact_id: Contact identifier
        interaction_type: Type of interaction (call, email, meeting, etc.)
        summary: Summary of interaction
        date: Interaction date (defaults to now)
        duration_seconds: Duration in seconds (for calls)

    Returns:
        Interaction record
    """
    interaction = Interaction(
        interaction_id="",
        contact_id=contact_id,
        interaction_type=interaction_type,
        summary=summary,
        date=date,
        duration_seconds=duration_seconds,
    )

    conn = sqlite3.connect(CONTACTS_DB)
    c = conn.cursor()
    c.execute('''INSERT INTO interactions
                 (interaction_id, contact_id, interaction_type, summary, date, duration_seconds, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (interaction.interaction_id, contact_id, interaction_type, summary,
               interaction.date, duration_seconds, interaction.created_at))
    conn.commit()
    conn.close()

    # Update contact's updated_at timestamp
    update_contact(contact_id, updated_at=datetime.now(timezone.utc).isoformat())

    return interaction


def get_contact_history(contact_id: str, limit: int = 50) -> List[Interaction]:
    """Get interaction history for a contact.

    Args:
        contact_id: Contact identifier
        limit: Max interactions to return

    Returns:
        List of interactions, most recent first
    """
    conn = sqlite3.connect(CONTACTS_DB)
    c = conn.cursor()
    c.execute('''SELECT * FROM interactions WHERE contact_id = ?
                 ORDER BY date DESC LIMIT ?''', (contact_id, limit))
    rows = c.fetchall()
    conn.close()

    interactions = []
    for row in rows:
        interactions.append(Interaction(
            interaction_id=row[0],
            contact_id=row[1],
            interaction_type=row[2],
            summary=row[3],
            date=row[4],
            duration_seconds=row[5],
            created_at=row[6],
        ))

    return interactions


def get_open_action_items(contact_id: str) -> List[Dict]:
    """Get open action items for a contact (from call_intel integration).

    Placeholder: would integrate with call_intel module.

    Args:
        contact_id: Contact identifier

    Returns:
        List of action item dicts
    """
    # TODO: Integrate with call_intel.get_action_items_for_contact()
    return []


# -- Relationship Scoring --

def get_relationship_score(contact_id: str, days_window: int = 180) -> float:
    """Score relationship strength based on interaction frequency and recency.

    Args:
        contact_id: Contact identifier
        days_window: Time window for scoring (default 180 days)

    Returns:
        Score 0.0 to 1.0
    """
    conn = sqlite3.connect(CONTACTS_DB)
    c = conn.cursor()

    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_window)).isoformat()

    c.execute('''SELECT COUNT(*) FROM interactions
                 WHERE contact_id = ? AND date > ?''',
              (contact_id, cutoff_date))
    recent_count = c.fetchone()[0]

    c.execute('SELECT COUNT(*) FROM interactions WHERE contact_id = ?',
              (contact_id,))
    total_count = c.fetchone()[0]

    conn.close()

    if total_count == 0:
        return 0.0

    # Score based on recency and frequency
    recency_score = min(recent_count / 10.0, 1.0)  # 0-1, saturates at 10 interactions
    frequency_score = min(total_count / 50.0, 1.0)  # 0-1, saturates at 50 total

    return (recency_score * 0.6 + frequency_score * 0.4)


# -- Pre-call Briefing --

def generate_briefing(contact_id: str) -> Optional[Briefing]:
    """Generate a pre-call briefing for a contact.

    Includes last interaction, open action items, relevant documents,
    and relationship history.

    Args:
        contact_id: Contact identifier

    Returns:
        Briefing record
    """
    contact = get_contact(contact_id)
    if not contact:
        return None

    # Get interaction history
    history = get_contact_history(contact_id, limit=10)
    history_dicts = [asdict(h) for h in history]

    # Get last interaction
    last_interaction = None
    if history:
        last_interaction = history[0].date

    # Get relationship score
    rel_score = get_relationship_score(contact_id)

    # Get open action items (placeholder)
    open_actions = get_open_action_items(contact_id)

    # Relevant documents (placeholder)
    relevant_docs = []

    briefing = Briefing(
        contact_id=contact_id,
        contact_name=contact.name,
        company=contact.company,
        role=contact.role,
        last_interaction=last_interaction,
        open_action_items=open_actions,
        relevant_documents=relevant_docs,
        relationship_strength=rel_score,
        interaction_history=history_dicts,
    )

    return briefing


def get_briefing_summary(contact_id: str) -> Optional[Dict]:
    """Get a brief summary for quick reference before call.

    Args:
        contact_id: Contact identifier

    Returns:
        Briefing dict or None
    """
    briefing = generate_briefing(contact_id)
    if not briefing:
        return None

    return {
        "contact_name": briefing.contact_name,
        "company": briefing.company,
        "role": briefing.role,
        "last_interaction": briefing.last_interaction,
        "relationship_strength": briefing.relationship_strength,
        "open_action_items_count": len(briefing.open_action_items),
        "recent_interactions_count": len(briefing.interaction_history),
        "summary": _brief_summary_text(briefing),
    }


def _brief_summary_text(briefing: Briefing) -> str:
    """Generate brief summary text for display."""
    lines = [f"{briefing.contact_name}"]

    if briefing.company:
        lines.append(f"Company: {briefing.company}")
    if briefing.role:
        lines.append(f"Role: {briefing.role}")

    rel_pct = int(briefing.relationship_strength * 100)
    lines.append(f"Relationship strength: {rel_pct}%")

    if briefing.last_interaction:
        lines.append(f"Last contact: {briefing.last_interaction[:10]}")

    if briefing.open_action_items:
        lines.append(f"Open action items: {len(briefing.open_action_items)}")

    return "\n".join(lines)


# -- Stats --

def get_contact_stats() -> Dict:
    """Get contact database statistics."""
    conn = sqlite3.connect(CONTACTS_DB)
    c = conn.cursor()

    c.execute('SELECT COUNT(*) FROM contacts')
    total_contacts = c.fetchone()[0]

    c.execute('SELECT COUNT(DISTINCT company) FROM contacts WHERE company IS NOT NULL')
    total_companies = c.fetchone()[0]

    c.execute('SELECT COUNT(*) FROM interactions')
    total_interactions = c.fetchone()[0]

    conn.close()

    return {
        "total_contacts": total_contacts,
        "total_companies": total_companies,
        "total_interactions": total_interactions,
    }


# Module-level convenience wrapper functions

def add_contact_wrapper(name, phone="", email="", company="", role="", tags=[], workspaces=[]):
    """Add a new contact.

    Args:
        name: Contact name
        phone: Phone number
        email: Email address
        company: Company name
        role: Job role
        tags: List of tags
        workspaces: Associated workspaces

    Returns:
        dict with contact data
    """
    contact = add_contact(name, phone or None, email or None, company or None, role or None, tags, workspaces)
    return contact.to_dict()


def search_contacts_wrapper(query):
    """Search for contacts.

    Args:
        query: Search query string

    Returns:
        list of contact dicts
    """
    contacts = search_contacts(query)
    return [c.to_dict() for c in contacts]


def generate_briefing_wrapper(contact_id):
    """Generate a pre-call briefing for a contact.

    Args:
        contact_id: Contact identifier

    Returns:
        dict with briefing data
    """
    briefing = generate_briefing(contact_id)
    if briefing:
        return briefing.to_dict()
    return None


def log_interaction_wrapper(contact_id, interaction_type="note", summary=""):
    """Log an interaction with a contact.

    Args:
        contact_id: Contact identifier
        interaction_type: Type of interaction
        summary: Interaction summary

    Returns:
        dict with interaction data
    """
    interaction = log_interaction(contact_id, interaction_type, summary)
    return interaction.to_dict()


def import_contacts_wrapper(contacts_list):
    """Import contacts from a list.

    Args:
        contacts_list: List of contact dicts or path to JSON file

    Returns:
        dict with import results
    """
    if isinstance(contacts_list, str):
        # It's a file path
        imported = import_contacts(contacts_list)
    else:
        # It's a list of dicts - save temporarily and import
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(contacts_list, f)
            temp_path = f.name
        try:
            imported = import_contacts(temp_path)
        finally:
            os.unlink(temp_path)

    return {
        'imported': imported,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }


def get_contact_history_wrapper(contact_id, limit=20):
    """Get interaction history for a contact.

    Args:
        contact_id: Contact identifier
        limit: Max interactions to return

    Returns:
        list of interaction dicts
    """
    interactions = get_contact_history(contact_id, limit)
    return [i.to_dict() for i in interactions]

"""
Call Intelligence Integration for VaultMind

Process call recordings and transcripts into structured knowledge:
  - Accept transcript text or audio file path
  - Transcript cleanup: remove filler words, normalize speaker labels
  - Structured summary generation
  - Sentiment analysis (positive/negative/neutral)
  - Topic extraction
  - Call metadata and tracking
  - Export ready for ChromaDB indexing

Storage: ~/.vaultmind/calls/
"""

import os
import json
import hashlib
import time
import re
from datetime import datetime, timezone
from typing import Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

CALLS_DIR = os.environ.get("VAULTMIND_CALLS_DIR", os.path.expanduser("~/.vaultmind/calls"))
METADATA_DIR = os.path.join(CALLS_DIR, "metadata")
TRANSCRIPTS_DIR = os.path.join(CALLS_DIR, "transcripts")

os.makedirs(CALLS_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)


class Sentiment(str, Enum):
    """Sentiment classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class ActionItem:
    """Represents an action item from a call."""
    description: str
    owner: Optional[str] = None
    due_date: Optional[str] = None
    priority: str = "medium"
    created_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CallRecord:
    """Complete call record with metadata and extracted data."""
    call_id: str
    transcript_text: str
    duration_seconds: Optional[int] = None
    date: Optional[str] = None
    participants: Optional[List[str]] = None
    related_workspace: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CallSummary:
    """Structured summary of a call."""
    call_id: str
    key_points: List[str]
    action_items: List[ActionItem]
    decisions_made: List[str]
    attendees: List[str]
    sentiment: str
    topics: List[str]
    duration_seconds: Optional[int] = None
    summary_text: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary, handling ActionItem objects."""
        data = asdict(self)
        data["action_items"] = [
            item.to_dict() if isinstance(item, ActionItem) else item
            for item in self.action_items
        ]
        return data


# -- Filler Words --

FILLER_WORDS = {
    "um", "uh", "uhh", "ah", "erm", "err",
    "like", "you know", "i mean", "basically",
    "actually", "literally", "honestly", "i think",
    "kind of", "sort of", "right", "okay", "so",
}


# -- Main Call Processing --

def process_transcript(
    transcript_text: Optional[str] = None,
    audio_file_path: Optional[str] = None,
    duration_seconds: Optional[int] = None,
    participants: Optional[List[str]] = None,
    workspace: Optional[str] = None,
) -> CallSummary:
    """Process a call transcript into structured knowledge.

    Args:
        transcript_text: Raw transcript text
        audio_file_path: Path to audio file (for future transcription)
        duration_seconds: Call duration in seconds
        participants: List of participant names
        workspace: Associated workspace

    Returns:
        CallSummary with extracted structure and insights
    """
    if not transcript_text:
        if audio_file_path:
            # For now, placeholder for future audio transcription
            transcript_text = f"[Audio transcription from {audio_file_path} not yet implemented]"
        else:
            raise ValueError("Either transcript_text or audio_file_path must be provided")

    # Generate call ID
    call_id = f"call_{int(time.time() * 1000)}_{hashlib.sha256(transcript_text.encode()).hexdigest()[:8]}"

    # Clean transcript
    cleaned = cleanup_transcript(transcript_text)

    # Extract components
    key_points = extract_key_points(cleaned)
    action_items = extract_action_items(cleaned)
    decisions = extract_decisions(cleaned)
    attendees = participants or extract_attendees(cleaned)
    sentiment = detect_sentiment(cleaned)
    topics = extract_topics(cleaned)

    # Create summary
    summary = CallSummary(
        call_id=call_id,
        key_points=key_points,
        action_items=action_items,
        decisions_made=decisions,
        attendees=attendees,
        sentiment=sentiment.value,
        topics=topics,
        duration_seconds=duration_seconds,
        summary_text=_generate_summary_text(key_points, decisions),
    )

    # Save call record
    call_record = CallRecord(
        call_id=call_id,
        transcript_text=cleaned,
        duration_seconds=duration_seconds,
        date=datetime.now(timezone.utc).isoformat(),
        participants=attendees,
        related_workspace=workspace,
    )

    _save_call_record(call_record, summary)

    return summary


def cleanup_transcript(text: str) -> str:
    """Remove filler words and normalize transcript.

    Args:
        text: Raw transcript text

    Returns:
        Cleaned transcript
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        # Remove speaker labels like "John:" or "[John]"
        line = re.sub(r"^\[?[\w\s]+[\]:]\s*", "", line)

        # Remove filler words (case-insensitive)
        for filler in FILLER_WORDS:
            line = re.sub(rf"\b{re.escape(filler)}\b", "", line, flags=re.IGNORECASE)

        # Clean up multiple spaces
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_key_points(text: str) -> List[str]:
    """Extract key points from transcript.

    Simple heuristics: sentences mentioning important keywords or
    following phrases like "key point", "important", "decision", etc.

    Args:
        text: Cleaned transcript text

    Returns:
        List of key point strings
    """
    key_points = []
    important_keywords = [
        "key", "important", "critical", "essential", "significant",
        "decided", "agreed", "approved", "confirmed", "planned",
    ]

    sentences = _split_sentences(text)

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in important_keywords):
            if len(sentence.split()) > 3:  # At least a few words
                key_points.append(sentence.strip())

    return key_points[:10]  # Top 10 key points


def extract_action_items(text: str) -> List[ActionItem]:
    """Extract action items from transcript.

    Looks for patterns like:
      - "will", "should", "need to", "have to", "must"
      - "action item:", "todo:", "follow up"

    Args:
        text: Cleaned transcript text

    Returns:
        List of ActionItem objects
    """
    action_items = []
    action_keywords = [
        "will", "should", "need to", "have to", "must",
        "follow up", "action item", "todo", "task",
    ]

    sentences = _split_sentences(text)

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in action_keywords):
            # Extract potential owner (words before "will", "should", etc.)
            owner = None
            for keyword in ["will", "should"]:
                if keyword in sentence_lower:
                    parts = sentence_lower.split(keyword)
                    if parts[0]:
                        owner = parts[0].strip().split()[-1]  # Last word before keyword
                    break

            action_items.append(ActionItem(
                description=sentence.strip(),
                owner=owner,
                priority="medium",
            ))

    return action_items[:15]  # Top 15 action items


def extract_decisions(text: str) -> List[str]:
    """Extract decisions made during call.

    Args:
        text: Cleaned transcript text

    Returns:
        List of decision strings
    """
    decisions = []
    decision_keywords = [
        "decided", "decision", "agreed", "approved",
        "will proceed", "we will", "we're going to",
    ]

    sentences = _split_sentences(text)

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in decision_keywords):
            decisions.append(sentence.strip())

    return decisions[:10]


def extract_attendees(text: str) -> List[str]:
    """Extract attendee names from transcript.

    Simple heuristic: words followed by colon at line start.

    Args:
        text: Transcript text (possibly uncleaned)

    Returns:
        List of attendee names
    """
    attendees = set()

    # Look for speaker labels like "John:" or "[John]"
    speaker_pattern = r"^\[?(\w+[\w\s]*?)[\]:]"
    for match in re.finditer(speaker_pattern, text, re.MULTILINE):
        name = match.group(1).strip()
        if len(name) < 50:  # Reasonable name length
            attendees.add(name)

    return list(attendees)


def detect_sentiment(text: str) -> Sentiment:
    """Analyze sentiment using simple word counting.

    Args:
        text: Transcript text

    Returns:
        Sentiment enum
    """
    positive_words = {
        "great", "excellent", "good", "perfect", "fantastic",
        "love", "amazing", "wonderful", "happy", "excited",
        "success", "accomplished", "achieved",
    }

    negative_words = {
        "bad", "terrible", "horrible", "awful", "hate",
        "problem", "issue", "concern", "worry", "difficult",
        "fail", "failed", "error", "risk", "uncertain",
    }

    text_lower = text.lower()
    words = text_lower.split()

    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)

    if positive_count > negative_count:
        return Sentiment.POSITIVE
    elif negative_count > positive_count:
        return Sentiment.NEGATIVE
    else:
        return Sentiment.NEUTRAL


def extract_topics(text: str) -> List[str]:
    """Extract topics discussed during call.

    Simple heuristic: common noun phrases.

    Args:
        text: Cleaned transcript text

    Returns:
        List of topic strings
    """
    # Extract 2-3 word phrases that appear multiple times
    word_pairs = {}

    words = text.lower().split()
    for i in range(len(words) - 1):
        pair = f"{words[i]} {words[i+1]}"
        # Skip short words and filler
        if len(words[i]) > 3 and len(words[i+1]) > 3:
            if pair not in FILLER_WORDS:
                word_pairs[pair] = word_pairs.get(pair, 0) + 1

    # Sort by frequency and return top topics
    topics = sorted(word_pairs.items(), key=lambda x: x[1], reverse=True)
    return [topic[0] for topic in topics[:10]]


# -- Storage & Retrieval --

def _save_call_record(call_record: CallRecord, summary: CallSummary):
    """Save call record to disk."""
    # Save metadata
    metadata_file = os.path.join(METADATA_DIR, f"{call_record.call_id}_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)

    # Save transcript
    transcript_file = os.path.join(TRANSCRIPTS_DIR, f"{call_record.call_id}_transcript.txt")
    with open(transcript_file, "w") as f:
        f.write(call_record.transcript_text)


def save_call_record(
    transcript_text: str,
    summary: Optional[CallSummary] = None,
    duration_seconds: Optional[int] = None,
    participants: Optional[List[str]] = None,
    workspace: Optional[str] = None,
) -> str:
    """Save a call record to disk.

    Args:
        transcript_text: The transcript text
        summary: Optional pre-computed CallSummary
        duration_seconds: Call duration
        participants: Attendees
        workspace: Related workspace

    Returns:
        Call ID
    """
    if not summary:
        summary = process_transcript(
            transcript_text=transcript_text,
            duration_seconds=duration_seconds,
            participants=participants,
            workspace=workspace,
        )

    call_record = CallRecord(
        call_id=summary.call_id,
        transcript_text=transcript_text,
        duration_seconds=duration_seconds,
        date=datetime.now(timezone.utc).isoformat(),
        participants=participants or summary.attendees,
        related_workspace=workspace,
    )

    _save_call_record(call_record, summary)
    return summary.call_id


def get_call_history(limit: int = 50, workspace: Optional[str] = None) -> List[dict]:
    """Get call history with metadata.

    Args:
        limit: Max results
        workspace: Filter by workspace

    Returns:
        List of call summaries
    """
    call_files = sorted(
        [f for f in os.listdir(METADATA_DIR) if f.endswith("_metadata.json")],
        key=lambda f: os.path.getmtime(os.path.join(METADATA_DIR, f)),
        reverse=True,
    )[:limit]

    results = []
    for filename in call_files:
        try:
            with open(os.path.join(METADATA_DIR, filename)) as f:
                call_data = json.load(f)
                results.append(call_data)
        except (json.JSONDecodeError, IOError):
            pass

    return results


def get_call_record(call_id: str) -> Optional[dict]:
    """Get a specific call record.

    Args:
        call_id: Call identifier

    Returns:
        Call summary with transcript
    """
    metadata_file = os.path.join(METADATA_DIR, f"{call_id}_metadata.json")
    transcript_file = os.path.join(TRANSCRIPTS_DIR, f"{call_id}_transcript.txt")

    result = {}

    if os.path.exists(metadata_file):
        try:
            with open(metadata_file) as f:
                result = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    if os.path.exists(transcript_file):
        try:
            with open(transcript_file) as f:
                result["transcript"] = f.read()
        except IOError:
            pass

    return result if result else None


# -- Helpers --

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitter on period, question, exclamation
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def _generate_summary_text(key_points: List[str], decisions: List[str]) -> str:
    """Generate summary text from key points and decisions."""
    summary = ""

    if key_points:
        summary += "Key Points:\n"
        for point in key_points[:3]:
            summary += f"  - {point}\n"

    if decisions:
        summary += "\nDecisions:\n"
        for decision in decisions[:3]:
            summary += f"  - {decision}\n"

    return summary.strip() if summary else "No summary available"


# Module-level convenience wrapper functions
# Note: process_transcript and get_call_history are already at module level
# This file already has the correct structure with module-level functions
# Additional wrappers provided below for consistency

def process_transcript_wrapper(transcript, participants=[], date=None, workspace=""):
    """Wrapper for process_transcript with explicit parameters.

    Args:
        transcript: Transcript text
        participants: List of participant names
        date: Call date
        workspace: Associated workspace

    Returns:
        dict with call summary data
    """
    result = process_transcript(
        transcript_text=transcript,
        participants=participants or None,
        workspace=workspace or None
    )
    return result.to_dict()


def get_call_history_wrapper(limit=20):
    """Wrapper for get_call_history.

    Args:
        limit: Max results to return

    Returns:
        list of call records
    """
    return get_call_history(limit=limit)

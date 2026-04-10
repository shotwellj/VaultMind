"""
VaultMind User Profile
Stores the owner's identity so VaultMind knows who it's working for.

When career/work questions come up and the vault is thin, this module
provides identity context and can fetch the user's LinkedIn profile
as a fallback source.

Usage:
    from user_profile import get_profile, update_profile, get_linkedin_context

Profile is stored at ~/.vaultmind/profile.json
"""

import json
import os
import re
import requests
from pathlib import Path
from typing import Optional


PROFILE_DIR = Path.home() / ".vaultmind" / "profile"
PROFILE_FILE = PROFILE_DIR / "profile.json"
LINKEDIN_CACHE = PROFILE_DIR / "linkedin_cache.json"

# Cache LinkedIn data for 24 hours
CACHE_TTL_SECONDS = 86400

DEFAULT_PROFILE = {
    "name": "",
    "linkedin_url": "",
    "email": "",
    "current_title": "",
    "location": "",
    "bio": "",
    "work_history_notes": "",
}

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _ensure_dir():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def get_profile() -> dict:
    """Load the user profile from disk."""
    _ensure_dir()
    if PROFILE_FILE.exists():
        try:
            with open(PROFILE_FILE, "r") as f:
                data = json.load(f)
            # Merge with defaults in case new fields were added
            merged = {**DEFAULT_PROFILE, **data}
            return merged
        except Exception:
            return dict(DEFAULT_PROFILE)
    return dict(DEFAULT_PROFILE)


def update_profile(updates: dict) -> dict:
    """Update specific fields in the user profile."""
    _ensure_dir()
    profile = get_profile()
    for key, value in updates.items():
        if key in DEFAULT_PROFILE:
            profile[key] = value
    with open(PROFILE_FILE, "w") as f:
        json.dump(profile, f, indent=2)
    return profile


def has_profile() -> bool:
    """Check if the user has set up their profile (at minimum, a name)."""
    profile = get_profile()
    return bool(profile.get("name", "").strip())


def get_identity_context() -> str:
    """Build a context string about the user's identity for the LLM.

    This gets injected into vault prompts so the LLM knows who the user is.
    """
    profile = get_profile()
    if not profile.get("name"):
        return ""

    parts = [f"USER IDENTITY: The person using VaultMind is {profile['name']}."]

    if profile.get("current_title"):
        parts.append(f"Current role: {profile['current_title']}.")
    if profile.get("location"):
        parts.append(f"Location: {profile['location']}.")
    if profile.get("linkedin_url"):
        parts.append(f"LinkedIn: {profile['linkedin_url']}.")
    if profile.get("bio"):
        parts.append(f"Bio: {profile['bio']}.")
    if profile.get("work_history_notes"):
        parts.append(f"Work history notes: {profile['work_history_notes']}.")

    return " ".join(parts)


# ---- LinkedIn Fallback ------------------------------------------------

def _is_career_question(query: str) -> bool:
    """Detect if a query is about the user's career, work history, or job roles."""
    career_patterns = [
        r"\b(career|work history|job history|resume|cv)\b",
        r"\b(my role|my title|my position|my job)\b",
        r"\b(promoted|promotion|raise|performance)\b",
        r"\b(worked at|work at|was I at|my time at)\b",
        r"\b(employer|company I worked|previous job|past job)\b",
        r"\b(experience at|tenure at|years at)\b",
        r"\b(what did I do at|what was I at|my career at)\b",
        r"\b(did I do at|did I work|what I did at)\b",
    ]
    q_lower = query.lower()
    for pattern in career_patterns:
        if re.search(pattern, q_lower):
            return True
    return False


def _get_cached_linkedin() -> Optional[str]:
    """Return cached LinkedIn text if fresh enough."""
    if not LINKEDIN_CACHE.exists():
        return None
    try:
        with open(LINKEDIN_CACHE, "r") as f:
            cache = json.load(f)
        import time
        if time.time() - cache.get("timestamp", 0) < CACHE_TTL_SECONDS:
            return cache.get("text", "")
    except Exception:
        pass
    return None


def _cache_linkedin(text: str):
    """Save LinkedIn text to cache."""
    _ensure_dir()
    import time
    with open(LINKEDIN_CACHE, "w") as f:
        json.dump({"text": text, "timestamp": time.time()}, f)


def fetch_linkedin_profile(linkedin_url: str) -> Optional[str]:
    """Fetch the user's own LinkedIn profile page and extract text.

    Uses the public profile page (no login required for public profiles).
    Returns extracted text or None if fetch fails.
    """
    if not linkedin_url:
        return None

    # Check cache first
    cached = _get_cached_linkedin()
    if cached:
        return cached

    # Clean URL
    url = linkedin_url.strip().rstrip("/")
    if not url.startswith("http"):
        url = "https://" + url

    try:
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.content, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "iframe", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Trim to reasonable size
        if len(text) > 8000:
            text = text[:8000]

        if text.strip():
            _cache_linkedin(text)
            return text

    except Exception as e:
        print(f"[UserProfile] LinkedIn fetch failed: {e}")

    return None


def get_linkedin_context(query: str) -> str:
    """If the query is career-related and we have a LinkedIn URL, fetch and return context.

    Returns a formatted context block or empty string.
    """
    if not _is_career_question(query):
        return ""

    profile = get_profile()
    linkedin_url = profile.get("linkedin_url", "")
    if not linkedin_url:
        return ""

    text = fetch_linkedin_profile(linkedin_url)
    if not text:
        return ""

    return (
        f"\n\nFROM USER'S LINKEDIN PROFILE ({linkedin_url}):\n"
        f"{text}\n\n"
        f"NOTE: This is the user's own LinkedIn profile. Use it as a factual source "
        f"for their career history, job titles, and work experience."
    )

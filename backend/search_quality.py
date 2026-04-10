"""
VaultMind Search Quality Filter
Phase 2 -- filters web results by source trustworthiness.

Not all web results are equal. A .gov source is more authoritative than
a random blog. A known content farm is less reliable than a law journal.

Trust Tiers:
  TIER 1 (Authoritative)  -- .gov, .edu, official standards bodies, law reviews
  TIER 2 (Reliable)       -- established news, major tech docs, Wikipedia
  TIER 3 (General)        -- most websites, forums, blogs with reputation
  TIER 4 (Low Quality)    -- content farms, SEO spam, known unreliable sources
  BLOCKED                 -- sites that should never appear in results

Each result gets a trust_score (0.0 - 1.0) and a quality_tier label.
Results below the minimum tier threshold are filtered out.

CROSS-PRODUCT NOTE:
  This module can also be used by AIR Blackbox to evaluate the quality
  of training data sources (Article 10 -- Data Governance).
"""

import re
import os
import json
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse


# ── Trust Tier Definitions ─────────────────────────────────────

TIER_1_PATTERNS = [
    # Government
    r"\.gov($|/)",
    r"\.gov\.\w{2}($|/)",  # .gov.uk, .gov.au, etc.
    r"europa\.eu",
    r"eur-lex\.europa\.eu",
    # Education
    r"\.edu($|/)",
    r"\.ac\.\w{2}($|/)",  # .ac.uk, .ac.jp
    # Standards bodies
    r"iso\.org",
    r"nist\.gov",
    r"w3\.org",
    r"ietf\.org",
    r"rfc-editor\.org",
    # Legal / Regulatory
    r"law\.cornell\.edu",
    r"supremecourt\.gov",
    r"courtlistener\.com",
    r"casetext\.com",
    r"westlaw\.com",
    r"lexisnexis\.com",
    r"ssrn\.com",
    # AI Governance specific
    r"artificialintelligenceact\.eu",
    r"aiact-explorer\.com",
    r"digital-strategy\.ec\.europa\.eu",
    r"airblackbox\.ai",  # Our own product
]

TIER_2_PATTERNS = [
    # Tech documentation
    r"docs\.\w+\.\w+",  # docs.python.org, docs.google.com
    r"developer\.\w+\.\w+",
    r"learn\.microsoft\.com",
    r"cloud\.google\.com",
    r"aws\.amazon\.com",
    # Reference
    r"wikipedia\.org",
    r"stackoverflow\.com",
    r"github\.com",
    r"arxiv\.org",
    r"scholar\.google\.com",
    # Established news / tech
    r"reuters\.com",
    r"apnews\.com",
    r"bbc\.com",
    r"nytimes\.com",
    r"theguardian\.com",
    r"techcrunch\.com",
    r"arstechnica\.com",
    r"theverge\.com",
    r"wired\.com",
    # AI / ML specific
    r"huggingface\.co",
    r"openai\.com",
    r"anthropic\.com",
    r"deepmind\.com",
    r"ollama\.com",
    r"pytorch\.org",
    r"tensorflow\.org",
]

BLOCKED_DOMAINS = [
    # Known content farms / low quality
    r"ehow\.com",
    r"wikihow\.com",  # Often low quality for professional topics
    r"quora\.com",  # User-generated, unreliable
    r"answers\.yahoo\.com",
    # SEO spam patterns
    r".*\.blogspot\.com",
    r".*\.weebly\.com",
    r".*-review\.com",
    r".*-reviews\.com",
    r"best-.*\.com",
    r"top\d+-.*\.com",
    # Paywalled with no useful snippets
    r"scribd\.com",
    r"slideshare\.net",
    # Ad/tracking
    r"doubleclick\.net",
    r"googlesyndication\.com",
]

# Compile patterns for performance
_tier1_re = [re.compile(p, re.IGNORECASE) for p in TIER_1_PATTERNS]
_tier2_re = [re.compile(p, re.IGNORECASE) for p in TIER_2_PATTERNS]
_blocked_re = [re.compile(p, re.IGNORECASE) for p in BLOCKED_DOMAINS]


# ── Quality Result ─────────────────────────────────────────────

@dataclass
class QualityResult:
    """Result of quality filtering."""
    filtered_results: list = field(default_factory=list)
    blocked_results: list = field(default_factory=list)
    total: int = 0
    kept: int = 0
    blocked: int = 0


# ── Core Functions ─────────────────────────────────────────────

def classify_domain(url: str) -> tuple[str, float]:
    """Classify a URL into a trust tier.

    Returns:
        (tier_name, trust_score) where tier is "tier1"/"tier2"/"tier3"/"tier4"/"blocked"
        and trust_score is 0.0-1.0
    """
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        return "tier4", 0.2

    if not domain:
        return "tier4", 0.2

    # Check blocked first
    for pattern in _blocked_re:
        if pattern.search(domain):
            return "blocked", 0.0

    # Check tier 1
    for pattern in _tier1_re:
        if pattern.search(domain):
            return "tier1", 1.0

    # Check tier 2
    for pattern in _tier2_re:
        if pattern.search(domain):
            return "tier2", 0.8

    # Heuristic signals for tier 3 vs tier 4
    score = 0.5  # Default tier 3

    # HTTPS is expected
    if url.startswith("https://"):
        score += 0.05

    # Short, clean domains are usually better
    parts = domain.split(".")
    if len(parts) <= 3:
        score += 0.05

    # Known TLDs that indicate quality
    quality_tlds = {".org", ".io", ".dev", ".app"}
    for tld in quality_tlds:
        if domain.endswith(tld):
            score += 0.05
            break

    # Very long domains or subdomains are often spam
    if len(domain) > 40:
        score -= 0.15

    # Numbers in domain (often autogenerated spam)
    if re.search(r'\d{3,}', domain):
        score -= 0.1

    if score >= 0.5:
        return "tier3", min(score, 0.7)
    else:
        return "tier4", max(score, 0.1)


def filter_results(
    results: list,
    min_tier: str = "tier3",
    min_score: float = 0.3,
) -> QualityResult:
    """Filter search results by quality tier.

    Args:
        results: List of SearchResult objects (or dicts with 'url' key)
        min_tier: Minimum acceptable tier ("tier1", "tier2", "tier3", "tier4")
        min_score: Minimum trust score to keep

    Returns:
        QualityResult with filtered and blocked results
    """
    tier_order = {"tier1": 1, "tier2": 2, "tier3": 3, "tier4": 4, "blocked": 5}
    min_tier_num = tier_order.get(min_tier, 3)

    quality = QualityResult(total=len(results))
    filtered = []
    blocked = []

    for result in results:
        # Support both SearchResult objects and dicts
        url = result.url if hasattr(result, "url") else result.get("url", "")
        tier, score = classify_domain(url)

        # Annotate the result
        if hasattr(result, "quality_tier"):
            result.quality_tier = tier
            result.trust_score = score
        elif isinstance(result, dict):
            result["quality_tier"] = tier
            result["trust_score"] = score

        tier_num = tier_order.get(tier, 4)
        if tier == "blocked" or tier_num > min_tier_num or score < min_score:
            blocked.append(result)
        else:
            filtered.append(result)

    # Sort by trust score (highest first)
    filtered.sort(
        key=lambda r: r.trust_score if hasattr(r, "trust_score") else r.get("trust_score", 0),
        reverse=True
    )

    quality.filtered_results = filtered
    quality.blocked_results = blocked
    quality.kept = len(filtered)
    quality.blocked = len(blocked)

    return quality


# ── User-Configurable Domain Lists ────────────────────────────

CUSTOM_LISTS_FILE = os.path.expanduser("~/.vaultmind/firewall/domain_lists.json")


def load_custom_domain_lists() -> dict:
    """Load user-configurable domain allow/block lists."""
    if os.path.exists(CUSTOM_LISTS_FILE):
        try:
            with open(CUSTOM_LISTS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"allowed": [], "blocked": []}


def save_custom_domain_lists(allowed: list[str] = None, blocked: list[str] = None):
    """Save custom domain lists."""
    current = load_custom_domain_lists()
    if allowed is not None:
        current["allowed"] = allowed
    if blocked is not None:
        current["blocked"] = blocked
    os.makedirs(os.path.dirname(CUSTOM_LISTS_FILE), exist_ok=True)
    with open(CUSTOM_LISTS_FILE, "w") as f:
        json.dump(current, f, indent=2)

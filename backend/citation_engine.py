"""
VaultMind Citation Engine
Phase 3 -- Source-linked claims with clickable document references.

Every answer VaultMind gives should tell the user WHERE it got the info.
This module:

  1. Extracts claims from the LLM response
  2. Matches each claim to the best source chunk
  3. Inserts inline citations like [1], [2]
  4. Builds a "Sources" footer with clickable links

CROSS-PRODUCT NOTE:
  Maps to AIR Blackbox Article 13 (Transparency) -- users must be able
  to trace AI outputs back to the data that produced them.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourceRef:
    """A single source reference."""
    index: int  # 1-based citation number
    label: str  # "Contract.pdf, Section 3"
    source_type: str  # "local" or "web"
    url: str = ""  # file path or web URL
    section: str = ""  # section header if available
    snippet: str = ""  # short excerpt from the source
    trust_tier: str = ""  # "tier1", "tier2", etc.


@dataclass
class CitedResponse:
    """A response with inline citations and a source footer."""
    text: str  # Response with [1], [2] inline citations
    sources: list = field(default_factory=list)  # List of SourceRef
    citation_count: int = 0
    uncited_claims: int = 0  # Claims we could not match to a source
    footer: str = ""  # Formatted source list for display

    def to_dict(self):
        return {
            "text": self.text,
            "sources": [
                {
                    "index": s.index,
                    "label": s.label,
                    "source_type": s.source_type,
                    "url": s.url,
                    "section": s.section,
                    "snippet": s.snippet[:100] if s.snippet else "",
                    "trust_tier": s.trust_tier,
                }
                for s in self.sources
            ],
            "citation_count": self.citation_count,
            "uncited_claims": self.uncited_claims,
            "footer": self.footer,
        }


# ── Claim Extraction ─────────────────────────────────────────

# Patterns that signal a factual claim worth citing
CLAIM_SIGNALS = [
    r'\b\d+(?:\.\d+)?%',  # Percentages
    r'\$[\d,]+(?:\.\d+)?',  # Dollar amounts
    r'\b\d{4}\b',  # Years
    r'\b(?:Article|Section|Clause)\s+\d+',  # Legal references
    r'\b(?:according to|per the|as stated in|based on)\b',  # Attribution phrases
    r'\b(?:requires?|mandates?|prohibits?|allows?)\b',  # Regulatory language
    r'\b(?:must|shall|should not)\b',  # Obligation language
]

# Sentence splitter (handles abbreviations reasonably well)
SENTENCE_SPLIT = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=\S)'
)


def extract_claims(response: str) -> list:
    """Break the response into sentences and identify which are factual claims.

    Returns a list of dicts:
        [{"text": "sentence", "is_claim": True/False, "start": 0, "end": 50}]
    """
    sentences = SENTENCE_SPLIT.split(response)
    claims = []
    pos = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Find the actual position in the original text
        start = response.find(sent, pos)
        if start == -1:
            start = pos
        end = start + len(sent)
        pos = end

        # Check if this sentence contains a factual claim
        is_claim = False
        for pattern in CLAIM_SIGNALS:
            if re.search(pattern, sent, re.IGNORECASE):
                is_claim = True
                break

        # Also flag sentences longer than 15 words that have specific terms
        if not is_claim and len(sent.split()) > 15:
            specific_words = [w for w in sent.split() if len(w) > 6 and w.isalpha()]
            if len(specific_words) >= 3:
                is_claim = True

        claims.append({
            "text": sent,
            "is_claim": is_claim,
            "start": start,
            "end": end,
        })

    return claims


# ── Source Matching ───────────────────────────────────────────

def _word_overlap_score(text_a: str, text_b: str) -> float:
    """Calculate word overlap ratio between two texts."""
    stops = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
             "have", "has", "had", "do", "does", "did", "will", "would",
             "to", "of", "in", "for", "on", "with", "at", "by", "from",
             "as", "and", "but", "or", "not", "this", "that", "it"}

    words_a = set(text_a.lower().split()) - stops
    words_b = set(text_b.lower().split()) - stops

    if not words_a or not words_b:
        return 0.0

    overlap = words_a & words_b
    return len(overlap) / min(len(words_a), len(words_b))


def _number_match_score(claim: str, source: str) -> float:
    """Check if specific numbers in the claim appear in the source."""
    nums = re.findall(r'\b\d[\d,.]*\b', claim)
    if not nums:
        return 0.0

    matched = sum(1 for n in nums if n in source)
    return matched / len(nums)


def match_claim_to_source(claim_text: str, sources: list) -> Optional[dict]:
    """Find the best matching source for a claim.

    Args:
        claim_text: The sentence/claim to match
        sources: List of dicts with keys: text, label, source_type, url, section, trust_tier

    Returns:
        Best matching source dict with added "score" key, or None
    """
    if not sources:
        return None

    best = None
    best_score = 0.0

    for source in sources:
        source_text = source.get("text", "")
        if not source_text:
            continue

        # Word overlap (60% weight)
        word_score = _word_overlap_score(claim_text, source_text)

        # Number match (30% weight)
        num_score = _number_match_score(claim_text, source_text)

        # Section header match bonus (10% weight)
        section = source.get("section", "").lower()
        claim_lower = claim_text.lower()
        section_bonus = 0.3 if section and any(
            w in claim_lower for w in section.split() if len(w) > 3
        ) else 0.0

        total = word_score * 0.6 + num_score * 0.3 + section_bonus * 0.1

        if total > best_score:
            best_score = total
            best = {**source, "score": total}

    # Only return a match if it passes the minimum threshold
    if best and best_score >= 0.15:
        return best

    return None


# ── Citation Insertion ────────────────────────────────────────

def insert_citations(response: str, sources: list) -> CitedResponse:
    """Main entry point: add inline citations and build source footer.

    Args:
        response: Raw LLM response text
        sources: List of source dicts with keys:
            text, label, source_type, url, section, trust_tier

    Returns:
        CitedResponse with cited text, source list, and footer
    """
    if not response.strip():
        return CitedResponse(text=response)

    claims = extract_claims(response)

    # Track which sources get cited (deduplicate by label)
    cited_sources = {}  # label -> SourceRef
    source_counter = 0
    claim_citations = []  # (claim_index, source_label)
    uncited = 0

    for claim in claims:
        if not claim["is_claim"]:
            claim_citations.append((claim, None))
            continue

        match = match_claim_to_source(claim["text"], sources)
        if match:
            label = match.get("label", "Unknown")
            if label not in cited_sources:
                source_counter += 1
                cited_sources[label] = SourceRef(
                    index=source_counter,
                    label=label,
                    source_type=match.get("source_type", "local"),
                    url=match.get("url", ""),
                    section=match.get("section", ""),
                    snippet=match.get("text", "")[:150],
                    trust_tier=match.get("trust_tier", ""),
                )
            claim_citations.append((claim, label))
        else:
            claim_citations.append((claim, None))
            uncited += 1

    # Rebuild the response with inline citations
    cited_text = response
    # Process claims in reverse order so positions stay correct
    for claim, label in reversed(claim_citations):
        if label and label in cited_sources:
            ref = cited_sources[label]
            marker = f" [{ref.index}]"
            # Insert citation marker at the end of the claim sentence
            end_pos = claim["end"]
            # If text ends with punctuation, insert before it
            while end_pos > claim["start"] and cited_text[end_pos - 1] in ".!?":
                end_pos -= 1
            cited_text = cited_text[:end_pos] + marker + cited_text[end_pos:]

    # Build the footer
    source_list = sorted(cited_sources.values(), key=lambda s: s.index)
    footer_lines = []
    if source_list:
        footer_lines.append("\n---\nSources:")
        for ref in source_list:
            if ref.source_type == "web" and ref.url:
                footer_lines.append(f"  [{ref.index}] {ref.label} - {ref.url}")
            elif ref.section:
                footer_lines.append(f"  [{ref.index}] {ref.label}, {ref.section}")
            else:
                footer_lines.append(f"  [{ref.index}] {ref.label}")
            if ref.trust_tier:
                footer_lines[-1] += f" (Trust: {ref.trust_tier})"

    footer = "\n".join(footer_lines)

    return CitedResponse(
        text=cited_text,
        sources=source_list,
        citation_count=len(source_list),
        uncited_claims=uncited,
        footer=footer,
    )


# ── Convenience Functions ─────────────────────────────────────

def cite_response(response: str, local_chunks: list = None, web_results: list = None) -> CitedResponse:
    """High-level convenience: merge local and web sources, then cite.

    Args:
        response: Raw LLM response
        local_chunks: List of local vault chunks, each a dict with:
            text, source (filename), section_header, char_start, char_end
        web_results: List of web results, each a dict with:
            title, url, snippet, trust_tier

    Returns:
        CitedResponse
    """
    sources = []

    # Convert local chunks to source format
    if local_chunks:
        for chunk in local_chunks:
            sources.append({
                "text": chunk.get("text", ""),
                "label": chunk.get("source", "Local document"),
                "source_type": "local",
                "url": chunk.get("source", ""),
                "section": chunk.get("section_header", ""),
                "trust_tier": "local",
            })

    # Convert web results to source format
    if web_results:
        for result in web_results:
            sources.append({
                "text": result.get("snippet", ""),
                "label": result.get("title", "Web source"),
                "source_type": "web",
                "url": result.get("url", ""),
                "section": "",
                "trust_tier": result.get("trust_tier", "tier3"),
            })

    return insert_citations(response, sources)


def format_sources_for_frontend(cited: CitedResponse) -> dict:
    """Format citation data for the VaultMind frontend.

    Returns a dict the frontend can use to render citation badges,
    tooltips, and a collapsible source panel.
    """
    return {
        "cited_text": cited.text,
        "source_panel": [
            {
                "number": s.index,
                "label": s.label,
                "type": s.source_type,
                "url": s.url,
                "section": s.section,
                "preview": s.snippet[:80] + "..." if len(s.snippet) > 80 else s.snippet,
                "tier": s.trust_tier,
            }
            for s in cited.sources
        ],
        "stats": {
            "total_citations": cited.citation_count,
            "uncited_claims": cited.uncited_claims,
            "coverage": f"{cited.citation_count}/{cited.citation_count + cited.uncited_claims}" if (cited.citation_count + cited.uncited_claims) > 0 else "N/A",
        },
        "footer_text": cited.footer,
    }

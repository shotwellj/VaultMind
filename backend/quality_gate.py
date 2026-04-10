"""
VaultMind Quality Gate
Phase 3 -- SLM verification pass on every response before the user sees it.

The Quality Gate runs a fast local model (Phi-3 or similar) as a second
opinion on every answer. It checks:

  1. GROUNDEDNESS  -- Are claims actually supported by the provided sources?
  2. RELEVANCE     -- Does the answer address the actual question asked?
  3. COMPLETENESS  -- Are there obvious gaps or missing information?
  4. CITATION       -- Are sources referenced where they should be?
  5. CONTRADICTION  -- Does anything conflict with the local knowledge base?

Each check returns a score. The overall confidence is the weighted average.
The result is a badge: HIGH / MEDIUM / LOW confidence.

This runs AFTER the main LLM generates the response, BEFORE the user sees it.
It adds ~1-2 seconds but prevents hallucinated garbage from reaching the user.

CROSS-PRODUCT NOTE:
  Maps to AIR Blackbox Article 15 (Robustness) -- ensuring AI outputs are
  accurate and grounded. The same verification engine can audit agent outputs.
"""

import re
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CheckType(str, Enum):
    GROUNDEDNESS = "groundedness"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CITATION = "citation"
    CONTRADICTION = "contradiction"


@dataclass
class QualityCheck:
    """Result of a single quality check."""
    check_type: CheckType
    score: float  # 0.0 - 1.0
    passed: bool
    detail: str = ""


@dataclass
class QualityVerdict:
    """Full quality gate verdict on a response."""
    confidence: ConfidenceLevel
    confidence_score: float  # 0.0 - 1.0
    checks: list = field(default_factory=list)
    issues: list = field(default_factory=list)  # Human-readable issue strings
    badge_text: str = ""  # "HIGH CONFIDENCE" etc.
    should_warn: bool = False  # True if user should see a warning
    verification_model: str = ""

    def to_dict(self):
        return {
            "confidence": self.confidence.value,
            "confidence_score": round(self.confidence_score, 3),
            "checks": [
                {
                    "type": c.check_type.value,
                    "score": round(c.score, 3),
                    "passed": c.passed,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
            "issues": self.issues,
            "badge_text": self.badge_text,
            "should_warn": self.should_warn,
            "verification_model": self.verification_model,
        }


# ── Check Weights ──────────────────────────────────────────────
# How much each check contributes to the overall confidence score

CHECK_WEIGHTS = {
    CheckType.GROUNDEDNESS: 0.35,
    CheckType.RELEVANCE: 0.25,
    CheckType.COMPLETENESS: 0.15,
    CheckType.CITATION: 0.15,
    CheckType.CONTRADICTION: 0.10,
}


# ── Fast Heuristic Checks (no LLM needed) ─────────────────────
# These run instantly and catch obvious problems before the LLM check.

def check_groundedness_heuristic(response: str, context: str) -> QualityCheck:
    """Check if the response uses information from the provided context.

    Heuristic: count how many key phrases from the response appear in the context.
    If the response invents lots of specifics not in the context, score drops.
    """
    if not context.strip():
        # No context provided -- can't verify groundedness
        return QualityCheck(
            check_type=CheckType.GROUNDEDNESS,
            score=0.5,
            passed=True,
            detail="No source context available for verification",
        )

    # Extract meaningful phrases from the response (3+ word sequences)
    response_words = response.lower().split()
    context_lower = context.lower()

    # Check for specific claims: numbers, names, dates
    number_pattern = re.compile(r'\b\d+(?:\.\d+)?(?:%|(?:\s*(?:million|billion|thousand|percent|years?|months?|days?|hours?)))?')
    response_numbers = set(number_pattern.findall(response.lower()))
    context_numbers = set(number_pattern.findall(context_lower))

    if response_numbers:
        grounded_numbers = response_numbers & context_numbers
        number_score = len(grounded_numbers) / len(response_numbers) if response_numbers else 1.0
    else:
        number_score = 1.0  # No numbers to verify

    # Check for key noun phrases (simple: words > 5 chars as proxy for specific terms)
    specific_terms = [w for w in response_words if len(w) > 5 and w.isalpha()]
    if specific_terms:
        grounded_terms = sum(1 for t in specific_terms if t in context_lower)
        term_score = min(1.0, grounded_terms / max(len(specific_terms) * 0.3, 1))
    else:
        term_score = 1.0

    # Check for job titles, roles, and proper nouns that must match the source
    # These are high-value claims that should NOT be hallucinated
    role_patterns = re.compile(
        r'\b(?:software engineer|data scientist|product manager|technical sourcer'
        r'|senior engineer|staff engineer|recruiter|analyst|designer|director'
        r'|vice president|manager|consultant|associate|partner|intern'
        r'|sourcer|coordinator|specialist|administrator|architect)\b',
        re.IGNORECASE
    )
    response_roles = set(m.group().lower() for m in role_patterns.finditer(response))
    context_roles = set(m.group().lower() for m in role_patterns.finditer(context))
    if response_roles:
        ungrounded_roles = response_roles - context_roles
        if ungrounded_roles:
            # Heavy penalty: the response invented job titles not in sources
            role_score = max(0.0, 1.0 - (len(ungrounded_roles) * 0.5))
        else:
            role_score = 1.0
    else:
        role_score = 1.0

    score = (number_score * 0.4 + term_score * 0.3 + role_score * 0.3)

    detail_parts = [f"Numbers: {number_score:.0%}", f"Terms: {term_score:.0%}", f"Roles: {role_score:.0%}"]
    if response_roles - context_roles:
        detail_parts.append(f"Ungrounded roles: {response_roles - context_roles}")

    return QualityCheck(
        check_type=CheckType.GROUNDEDNESS,
        score=score,
        passed=score >= 0.4,
        detail=", ".join(detail_parts),
    )


def check_relevance_heuristic(response: str, question: str) -> QualityCheck:
    """Check if the response addresses the actual question.

    Heuristic: check keyword overlap between question and response.
    Also check if the response contains question-type indicators
    (e.g., question asks "when" -- response should have dates/times).
    """
    q_words = set(question.lower().split())
    r_words = set(response.lower().split())

    # Remove stop words
    stops = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
             "have", "has", "had", "do", "does", "did", "will", "would", "could",
             "should", "may", "might", "can", "shall", "to", "of", "in", "for",
             "on", "with", "at", "by", "from", "as", "into", "through", "during",
             "before", "after", "above", "below", "between", "about", "this", "that",
             "these", "those", "i", "me", "my", "you", "your", "he", "she", "it",
             "we", "they", "what", "which", "who", "whom", "when", "where", "why",
             "how", "and", "but", "or", "not", "no", "if", "then", "so"}

    q_meaningful = q_words - stops
    r_meaningful = r_words - stops

    if not q_meaningful:
        return QualityCheck(
            check_type=CheckType.RELEVANCE,
            score=0.8,
            passed=True,
            detail="Short query, assuming relevant",
        )

    overlap = q_meaningful & r_meaningful
    overlap_ratio = len(overlap) / len(q_meaningful) if q_meaningful else 0

    # Check question type signals
    q_lower = question.lower()
    bonus = 0.0
    if "when" in q_lower and re.search(r'\b\d{4}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', response.lower()):
        bonus = 0.1
    elif "how many" in q_lower and re.search(r'\b\d+\b', response):
        bonus = 0.1
    elif "who" in q_lower and re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', response):
        bonus = 0.1

    score = min(1.0, overlap_ratio * 1.5 + bonus)

    return QualityCheck(
        check_type=CheckType.RELEVANCE,
        score=score,
        passed=score >= 0.3,
        detail=f"Question keyword overlap: {overlap_ratio:.0%}",
    )


def check_completeness_heuristic(response: str, question: str) -> QualityCheck:
    """Check if the response seems complete or suspiciously short/truncated."""
    word_count = len(response.split())

    # Very short responses to non-trivial questions are suspicious
    q_word_count = len(question.split())
    if q_word_count > 10 and word_count < 15:
        return QualityCheck(
            check_type=CheckType.COMPLETENESS,
            score=0.3,
            passed=False,
            detail=f"Response ({word_count} words) seems too short for the question ({q_word_count} words)",
        )

    # Check for truncation signals
    truncation_signals = [
        response.rstrip().endswith("..."),
        response.rstrip().endswith(","),
        response.rstrip().endswith(" and"),
        response.rstrip().endswith(" or"),
    ]
    if any(truncation_signals):
        return QualityCheck(
            check_type=CheckType.COMPLETENESS,
            score=0.5,
            passed=True,
            detail="Response may be truncated",
        )

    # Check for "I don't know" type non-answers
    hedges = ["i don't know", "i'm not sure", "i cannot", "i can't determine",
              "no relevant information", "not enough information", "unable to"]
    response_lower = response.lower()
    hedge_count = sum(1 for h in hedges if h in response_lower)
    if hedge_count >= 2:
        return QualityCheck(
            check_type=CheckType.COMPLETENESS,
            score=0.4,
            passed=True,
            detail="Response contains multiple hedging phrases",
        )

    score = min(1.0, 0.5 + (word_count / 200) * 0.5)

    return QualityCheck(
        check_type=CheckType.COMPLETENESS,
        score=score,
        passed=True,
        detail=f"Response length: {word_count} words",
    )


def check_citation_heuristic(response: str, sources: list) -> QualityCheck:
    """Check if the response cites its sources."""
    if not sources:
        return QualityCheck(
            check_type=CheckType.CITATION,
            score=0.7,
            passed=True,
            detail="No sources to cite",
        )

    # Look for citation patterns in the response
    citation_patterns = [
        r'\[LOCAL[:\s]',
        r'\[WEB[:\s]',
        r'\[Source[:\s#]',
        r'according to',
        r'based on',
        r'from the document',
        r'the (?:document|file|source) (?:states|says|mentions|indicates)',
        r'\.pdf',
        r'\.docx?',
        r'https?://',
    ]

    citations_found = 0
    for pattern in citation_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            citations_found += 1

    # Score based on how many citation signals we found relative to sources
    if len(sources) > 0:
        expected = min(len(sources), 3)  # Expect at least some citations
        score = min(1.0, citations_found / expected) if expected > 0 else 0.5
    else:
        score = 0.7

    return QualityCheck(
        check_type=CheckType.CITATION,
        score=score,
        passed=score >= 0.3,
        detail=f"Found {citations_found} citation signals for {len(sources)} sources",
    )


# ── Contradiction Detection ────────────────────────────────────

def check_contradictions(response: str, local_context: str, web_context: str = "") -> QualityCheck:
    """Detect contradictions between the response and sources, or between sources.

    This is the heuristic version. Looks for:
    - Conflicting numbers (different dollar amounts, dates, percentages)
    - Negation conflicts ("is" vs "is not", "allows" vs "prohibits")
    """
    issues = []

    # Extract numbers from each source
    num_pattern = re.compile(r'(?:\$\s?)?\d[\d,]*(?:\.\d+)?(?:\s*%)?')

    local_nums = set(num_pattern.findall(local_context.lower())) if local_context else set()
    web_nums = set(num_pattern.findall(web_context.lower())) if web_context else set()
    response_nums = set(num_pattern.findall(response.lower()))

    # Check if response numbers exist in at least one source
    if response_nums and (local_nums or web_nums):
        all_source_nums = local_nums | web_nums
        ungrounded = response_nums - all_source_nums
        if ungrounded and len(ungrounded) > len(response_nums) * 0.5:
            issues.append(f"Response contains numbers not found in sources: {', '.join(list(ungrounded)[:3])}")

    # Check for date conflicts between local and web
    date_pattern = re.compile(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b', re.IGNORECASE)
    local_dates = set(date_pattern.findall(local_context)) if local_context else set()
    web_dates = set(date_pattern.findall(web_context)) if web_context else set()

    if local_dates and web_dates:
        # If both have dates, check if they conflict on the same topic
        # (Simple heuristic: flag if dates differ significantly)
        if local_dates != web_dates and local_dates and web_dates:
            issues.append("Local documents and web sources contain different dates. Verify which is current.")

    # Check for negation conflicts
    neg_pairs = [
        (r'\bis\b', r'\bis not\b'),
        (r'\ballows?\b', r'\bprohibits?\b'),
        (r'\brequired\b', r'\boptional\b'),
        (r'\bmust\b', r'\bmay\b'),
        (r'\bapproved\b', r'\bdenied\b'),
        (r'\bcompliant\b', r'\bnon-?compliant\b'),
    ]

    if local_context and web_context:
        for pos, neg in neg_pairs:
            local_has_pos = bool(re.search(pos, local_context, re.IGNORECASE))
            web_has_neg = bool(re.search(neg, web_context, re.IGNORECASE))
            local_has_neg = bool(re.search(neg, local_context, re.IGNORECASE))
            web_has_pos = bool(re.search(pos, web_context, re.IGNORECASE))

            if (local_has_pos and web_has_neg) or (local_has_neg and web_has_pos):
                issues.append(f"Possible contradiction between local docs and web results (conflicting language detected)")
                break

    score = 1.0 - (len(issues) * 0.3)
    score = max(0.0, min(1.0, score))

    return QualityCheck(
        check_type=CheckType.CONTRADICTION,
        score=score,
        passed=len(issues) == 0,
        detail="; ".join(issues) if issues else "No contradictions detected",
    )


# ── LLM-Based Verification (optional, deeper check) ───────────

VERIFICATION_PROMPT = """You are a fact-checking assistant. Given a QUESTION, SOURCES, and a RESPONSE, evaluate the response quality.

Rate each dimension 1-5:
- GROUNDED: Are claims supported by the sources? (1=fabricated, 5=fully grounded)
- RELEVANT: Does it answer the actual question? (1=off-topic, 5=directly answers)
- COMPLETE: Are there obvious gaps? (1=missing critical info, 5=thorough)
- CITED: Are sources referenced? (1=no citations, 5=every claim cited)

Reply ONLY with JSON, no other text:
{{"grounded": N, "relevant": N, "complete": N, "cited": N, "issues": ["issue1", "issue2"]}}

QUESTION: {question}

SOURCES (abbreviated):
{sources}

RESPONSE TO VERIFY:
{response}"""


def verify_with_llm(
    response: str,
    question: str,
    context: str,
    model: str = "phi3",
) -> Optional[dict]:
    """Run LLM-based verification (slower but more thorough).

    Returns None if LLM verification fails or is unavailable.
    """
    try:
        import ollama

        # Truncate for the verification model's context window
        ctx_truncated = context[:2000] if len(context) > 2000 else context
        resp_truncated = response[:1500] if len(response) > 1500 else response

        prompt = VERIFICATION_PROMPT.format(
            question=question,
            sources=ctx_truncated,
            response=resp_truncated,
        )

        result = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 200},
        )

        raw = result["message"]["content"].strip()

        # Try to extract JSON from the response
        json_match = re.search(r'\{[^}]+\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "grounded": data.get("grounded", 3) / 5.0,
                "relevant": data.get("relevant", 3) / 5.0,
                "complete": data.get("complete", 3) / 5.0,
                "cited": data.get("cited", 3) / 5.0,
                "issues": data.get("issues", []),
            }
    except Exception as e:
        print(f"[QualityGate] LLM verification failed: {e}")

    return None


# ── Main Quality Gate Function ─────────────────────────────────

def run_quality_gate(
    response: str,
    question: str,
    context: str = "",
    local_context: str = "",
    web_context: str = "",
    sources: list = None,
    use_llm: bool = False,
    llm_model: str = "phi3",
) -> QualityVerdict:
    """Run the full quality gate on a response.

    Args:
        response: The LLM-generated response to verify
        question: The original user question
        context: Full combined context (local + web)
        local_context: Just the local vault context
        web_context: Just the web search context
        sources: List of source labels
        use_llm: Whether to run the slower LLM verification too
        llm_model: Which model to use for LLM verification

    Returns:
        QualityVerdict with confidence level, checks, and issues
    """
    if sources is None:
        sources = []

    checks = []
    issues = []

    # Run heuristic checks (instant)
    checks.append(check_groundedness_heuristic(response, context))
    checks.append(check_relevance_heuristic(response, question))
    checks.append(check_completeness_heuristic(response, question))
    checks.append(check_citation_heuristic(response, sources))
    checks.append(check_contradictions(response, local_context, web_context))

    # Optionally run LLM verification (1-2 seconds)
    verification_model = ""
    if use_llm:
        llm_result = verify_with_llm(response, question, context, llm_model)
        if llm_result:
            verification_model = llm_model
            # Blend LLM scores with heuristic scores (LLM gets 40% weight)
            for check in checks:
                llm_key = {
                    CheckType.GROUNDEDNESS: "grounded",
                    CheckType.RELEVANCE: "relevant",
                    CheckType.COMPLETENESS: "complete",
                    CheckType.CITATION: "cited",
                }.get(check.check_type)
                if llm_key and llm_key in llm_result:
                    check.score = check.score * 0.6 + llm_result[llm_key] * 0.4

            if llm_result.get("issues"):
                issues.extend(llm_result["issues"])

    # Collect issues from failed checks
    for check in checks:
        if not check.passed and check.detail:
            issues.append(f"[{check.check_type.value}] {check.detail}")

    # Calculate weighted confidence score
    total_weight = sum(CHECK_WEIGHTS.values())
    weighted_score = sum(
        check.score * CHECK_WEIGHTS.get(check.check_type, 0.1)
        for check in checks
    ) / total_weight

    # Determine confidence level
    if weighted_score >= 0.7:
        confidence = ConfidenceLevel.HIGH
        badge = "HIGH CONFIDENCE"
        should_warn = False
    elif weighted_score >= 0.45:
        confidence = ConfidenceLevel.MEDIUM
        badge = "MEDIUM CONFIDENCE"
        should_warn = False
    else:
        confidence = ConfidenceLevel.LOW
        badge = "LOW CONFIDENCE -- verify independently"
        should_warn = True

    return QualityVerdict(
        confidence=confidence,
        confidence_score=weighted_score,
        checks=checks,
        issues=issues,
        badge_text=badge,
        should_warn=should_warn,
        verification_model=verification_model,
    )

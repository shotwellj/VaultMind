"""
VaultMind Query Intelligence Engine v1
Classifies every query before anything else happens:
  - Intent detection (research, draft, summarize, compare, analyze, chat)
  - Complexity scoring (routes to fast SLM vs thorough large model)
  - Prompt template selection (each task type gets a purpose-built prompt)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Intent Types ─────────────────────────────────────────────

class QueryIntent(str, Enum):
    RESEARCH   = "research"     # Find information, look up facts, investigate
    DRAFT      = "draft"        # Write something new (email, memo, brief, etc.)
    SUMMARIZE  = "summarize"    # Condense a document or conversation
    COMPARE    = "compare"      # Side-by-side analysis of two+ things
    ANALYZE    = "analyze"      # Deep dive into a topic, risk assessment, etc.
    EXTRACT    = "extract"      # Pull specific data points from documents
    CHAT       = "chat"         # General conversation, follow-up, clarification
    ACTION     = "action"       # Agentic command (create file, schedule, etc.)


class ComplexityLevel(str, Enum):
    LOW    = "low"      # Quick factual lookup, simple question
    MEDIUM = "medium"   # Moderate reasoning, multi-step but straightforward
    HIGH   = "high"     # Complex analysis, legal reasoning, multi-document


@dataclass
class QueryClassification:
    intent: QueryIntent
    complexity: ComplexityLevel
    confidence: float           # 0.0 to 1.0
    recommended_model: str      # model name suggestion
    prompt_template: str        # which template to use
    reasoning: str              # why this classification
    needs_web: bool = False     # should we search the web?
    needs_vault: bool = True    # should we search local docs?
    extract_targets: list = field(default_factory=list)  # for EXTRACT intent


# ── Intent Detection Patterns ────────────────────────────────

INTENT_PATTERNS = {
    QueryIntent.DRAFT: [
        r"\b(write|draft|compose|create|prepare)\s+(me\s+)?(a|an|the|my)?\s*(email|memo|letter|brief|response|reply|message|document|report|summary|outline|proposal|contract|agreement)",
        r"\b(help me write|can you draft|put together|write me)\b",
        r"\b(rewrite|rephrase|revise|edit)\s+(this|the|my)\b",
    ],
    QueryIntent.SUMMARIZE: [
        r"\b(summarize|summary|summarise|tldr|tl;dr|brief me|give me the gist)\b",
        r"\b(what are the key points|what does .+ say|break down)\b",
        r"\b(condense|shorten|boil down)\b",
        r"\b(overview of|recap|highlights)\b",
    ],
    QueryIntent.COMPARE: [
        r"\b(compare|comparison|versus|vs\.?|difference between|how does .+ differ)\b",
        r"\b(which is better|pros and cons|side by side|weigh|trade.?offs?)\b",
        r"\b(contrast|distinguish between|similarities and differences)\b",
    ],
    QueryIntent.ANALYZE: [
        r"\b(analyze|analyse|analysis|assess|evaluate|review|examine|investigate)\b",
        r"\b(risk assessment|due diligence|deep dive|implications|impact)\b",
        r"\b(what are the risks|strengths and weaknesses|swot)\b",
        r"\b(why did|how did|what caused|root cause)\b",
        r"\b(interpret|explain why|what does this mean)\b",
    ],
    QueryIntent.EXTRACT: [
        r"\b(extract|pull out|find all|list all|get me the)\b",
        r"\b(what are the dates|who are the parties|what is the amount)\b",
        r"\b(names mentioned|key terms|deadlines|obligations)\b",
        r"\b(how much|what percentage|what number)\b",
    ],
    QueryIntent.RESEARCH: [
        r"\b(research|look up|find out|what is|who is|when did|where is)\b",
        r"\b(tell me about|what do you know about|information on)\b",
        r"\b(search for|look into|investigate|dig into)\b",
        r"\b(current status|latest|recent|update on)\b",
    ],
    QueryIntent.ACTION: [
        r"\b(schedule|create a task|set a reminder|move file|organize|file this)\b",
        r"\b(send email|create document|tag|log time|check conflicts)\b",
        r"\b(prepare for|set up|configure|automate)\b",
    ],
}

# ── Complexity Signals ───────────────────────────────────────

HIGH_COMPLEXITY_SIGNALS = [
    r"\b(legal|compliance|regulatory|statute|regulation|liability)\b",
    r"\b(multi.?party|cross.?reference|across .+ documents|all .+ files)\b",
    r"\b(comprehensive|thorough|detailed|in.?depth|exhaustive)\b",
    r"\b(implications|ramifications|downstream effects)\b",
    r"\b(strategy|strategic|long.?term|roadmap)\b",
    r"\b(precedent|case law|jurisprudence)\b",
    r"(and also|additionally|furthermore|moreover).+(and also|additionally|furthermore|moreover)",
]

LOW_COMPLEXITY_SIGNALS = [
    r"^(what is|who is|when|where|how much|yes or no)\b",
    r"^(is|are|do|does|did|can|will|should)\s+\w+\s*\??\s*$",
    r"\b(quick|briefly|short answer|just tell me|simple)\b",
    r"^[^.?!]{1,60}[.?!]?\s*$",  # Very short queries (under 60 chars)
]


# ── Model Recommendations ────────────────────────────────────

# Maps (intent, complexity) to recommended model
# These are Ollama model names -- users can override
MODEL_RECOMMENDATIONS = {
    # Low complexity -- use fast small model
    (QueryIntent.CHAT, ComplexityLevel.LOW):       "phi3",
    (QueryIntent.RESEARCH, ComplexityLevel.LOW):    "phi3",
    (QueryIntent.EXTRACT, ComplexityLevel.LOW):     "phi3",
    (QueryIntent.SUMMARIZE, ComplexityLevel.LOW):   "qwen2.5",

    # Medium complexity -- balanced model
    (QueryIntent.CHAT, ComplexityLevel.MEDIUM):     "mistral",
    (QueryIntent.RESEARCH, ComplexityLevel.MEDIUM):  "mistral",
    (QueryIntent.DRAFT, ComplexityLevel.MEDIUM):     "mistral",
    (QueryIntent.SUMMARIZE, ComplexityLevel.MEDIUM): "qwen2.5",
    (QueryIntent.EXTRACT, ComplexityLevel.MEDIUM):   "mistral",
    (QueryIntent.COMPARE, ComplexityLevel.MEDIUM):   "mistral",
    (QueryIntent.ANALYZE, ComplexityLevel.MEDIUM):   "mistral",

    # High complexity -- use thorough large model
    (QueryIntent.ANALYZE, ComplexityLevel.HIGH):    "llama3.1",
    (QueryIntent.COMPARE, ComplexityLevel.HIGH):    "llama3.1",
    (QueryIntent.DRAFT, ComplexityLevel.HIGH):      "llama3.1",
    (QueryIntent.RESEARCH, ComplexityLevel.HIGH):   "llama3.1",
    (QueryIntent.CHAT, ComplexityLevel.HIGH):       "llama3.1",
    (QueryIntent.SUMMARIZE, ComplexityLevel.HIGH):  "qwen2.5",
    (QueryIntent.EXTRACT, ComplexityLevel.HIGH):    "llama3.1",

    # Actions always go through the action pipeline
    (QueryIntent.ACTION, ComplexityLevel.LOW):      "qwen2.5",
    (QueryIntent.ACTION, ComplexityLevel.MEDIUM):   "qwen2.5",
    (QueryIntent.ACTION, ComplexityLevel.HIGH):     "qwen2.5",
}

# Fallback if model not available
DEFAULT_MODEL_FALLBACK = "mistral"


# ── Prompt Templates ─────────────────────────────────────────

PROMPT_TEMPLATES = {
    QueryIntent.RESEARCH: """You are a thorough research assistant. Answer the user's question with accuracy and depth.

INSTRUCTIONS:
- Cite your sources clearly: [Source: filename, section] for local docs, [Web: url] for web results
- If you are uncertain, say so explicitly rather than guessing
- Organize findings with clear structure
- Distinguish between facts and interpretations
- If the local documents contain relevant information, prioritize that over general knowledge

{context}

USER QUESTION: {query}""",

    QueryIntent.DRAFT: """You are a skilled writer and editor. Help the user create the requested document.

INSTRUCTIONS:
- Match the appropriate tone and formality for the document type
- Use clear, professional language unless instructed otherwise
- Structure the document with appropriate headers and sections
- If referencing local documents for context, note which ones informed your draft
- Ask clarifying questions if critical details are missing (audience, tone, length)

CONTEXT FROM LOCAL DOCUMENTS:
{context}

USER REQUEST: {query}""",

    QueryIntent.SUMMARIZE: """You are a precise summarizer. Condense the provided content while preserving all critical information.

INSTRUCTIONS:
- Lead with the most important takeaway
- Preserve key facts, numbers, dates, and names exactly
- Note any ambiguities or gaps in the source material
- Keep the summary proportional to the source length (aim for 20-30% of original)
- Tag each point with its source: [Source: filename]

CONTENT TO SUMMARIZE:
{context}

USER REQUEST: {query}""",

    QueryIntent.COMPARE: """You are an analytical assistant specializing in side-by-side comparisons.

INSTRUCTIONS:
- Create a structured comparison with clear categories
- Note similarities AND differences
- Highlight which option is stronger in each category
- Flag any missing information that would affect the comparison
- Present a balanced view -- do not favor one side without evidence
- Cite sources for each claim: [Source: filename]

MATERIALS FOR COMPARISON:
{context}

USER REQUEST: {query}""",

    QueryIntent.ANALYZE: """You are a deep analytical thinker. Provide thorough analysis with clear reasoning.

INSTRUCTIONS:
- Start with a clear thesis or finding
- Support each claim with evidence from the provided documents
- Consider multiple perspectives and counterarguments
- Identify risks, assumptions, and limitations
- Provide actionable recommendations where appropriate
- Cite all sources: [Source: filename] for local docs

ANALYSIS MATERIALS:
{context}

USER REQUEST: {query}""",

    QueryIntent.EXTRACT: """You are a precise data extraction assistant. Pull exactly the information requested.

INSTRUCTIONS:
- Extract only what was asked for -- no editorializing
- Present extracted data in a clean, structured format
- Note the exact source and location for each extracted item: [Source: filename, chunk N]
- If a requested item is not found, explicitly state "Not found in provided documents"
- If data is ambiguous, present all variants with context

DOCUMENTS TO EXTRACT FROM:
{context}

USER REQUEST: {query}""",

    QueryIntent.CHAT: """You are VaultMind, a helpful AI assistant with access to the user's private document vault.

INSTRUCTIONS:
- Be conversational but informative
- Reference relevant documents when they add value
- If the user's question relates to their indexed documents, search and cite them
- Keep responses focused and concise unless depth is requested

{context}

USER: {query}""",

    QueryIntent.ACTION: """You are VaultMind in agent mode. Parse the user's command and determine the right action.

INSTRUCTIONS:
- Identify the specific action requested
- Determine what information is needed to execute it
- For document creation: identify type, audience, tone, and key content
- For scheduling: identify time, participants, and purpose
- Always confirm before executing irreversible actions

CONTEXT:
{context}

USER COMMAND: {query}""",
}


# ── Core Classification Function ─────────────────────────────

def classify_query(
    query: str,
    conversation_history: list = None,
    available_models: list = None,
) -> QueryClassification:
    """
    Classify a user query by intent, complexity, and route to the right model.

    This runs locally with regex patterns -- no LLM call needed.
    Classification happens in <1ms, not <1 second.

    Args:
        query: The user's raw query text
        conversation_history: Previous messages for context (optional)
        available_models: List of models installed in Ollama (optional)

    Returns:
        QueryClassification with intent, complexity, model recommendation, and template
    """
    q_lower = query.lower().strip()

    # ── Step 1: Detect intent ────────────────────────────────
    intent_scores: dict[QueryIntent, float] = {}

    for intent, patterns in INTENT_PATTERNS.items():
        score = 0.0
        for pattern in patterns:
            matches = re.findall(pattern, q_lower)
            if matches:
                score += len(matches) * 0.3
        intent_scores[intent] = min(score, 1.0)

    # Check conversation context for follow-up detection
    is_followup = False
    if conversation_history and len(conversation_history) >= 2:
        # Short queries after a conversation are likely follow-ups
        if len(q_lower.split()) <= 8:
            is_followup = True

    # Pick the highest-scoring intent
    if intent_scores:
        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]
    else:
        best_intent = QueryIntent.CHAT
        best_score = 0.0

    # If no strong signal, default to CHAT (or RESEARCH if it looks like a question)
    if best_score < 0.2:
        if q_lower.rstrip("?").endswith("?") or q_lower.startswith(("what", "who", "when", "where", "how", "why", "is", "are", "do", "does", "can", "will")):
            best_intent = QueryIntent.RESEARCH
            best_score = 0.4
        elif is_followup:
            best_intent = QueryIntent.CHAT
            best_score = 0.5
        else:
            best_intent = QueryIntent.CHAT
            best_score = 0.3

    # ── Step 2: Score complexity ─────────────────────────────
    complexity_score = 0.5  # start at medium

    # Check for high complexity signals
    for pattern in HIGH_COMPLEXITY_SIGNALS:
        if re.search(pattern, q_lower):
            complexity_score += 0.15

    # Check for low complexity signals
    for pattern in LOW_COMPLEXITY_SIGNALS:
        if re.search(pattern, q_lower):
            complexity_score -= 0.2

    # Word count affects complexity
    word_count = len(q_lower.split())
    if word_count > 40:
        complexity_score += 0.15
    elif word_count < 8:
        complexity_score -= 0.15

    # Multiple questions = higher complexity
    question_marks = q_lower.count("?")
    if question_marks > 1:
        complexity_score += 0.1 * (question_marks - 1)

    # Clamp and categorize
    complexity_score = max(0.0, min(1.0, complexity_score))
    if complexity_score < 0.35:
        complexity = ComplexityLevel.LOW
    elif complexity_score < 0.65:
        complexity = ComplexityLevel.MEDIUM
    else:
        complexity = ComplexityLevel.HIGH

    # ── Step 3: Model recommendation ─────────────────────────
    recommended = MODEL_RECOMMENDATIONS.get(
        (best_intent, complexity),
        DEFAULT_MODEL_FALLBACK
    )

    # If the recommended model isn't available, fall back
    if available_models and recommended not in available_models:
        # Try to find a suitable alternative
        preferred_order = ["mistral", "llama3.1", "qwen2.5", "phi3", "gemma2", "deepseek-r1"]
        for fallback in preferred_order:
            if fallback in available_models:
                recommended = fallback
                break
        else:
            # Use whatever is available
            if available_models:
                recommended = available_models[0]

    # ── Step 4: Web search decision ──────────────────────────
    needs_web = False
    web_patterns = [
        r"\b(latest|recent|current|today|this week|this month|2024|2025|2026)\b",
        r"\b(news|update|announcement|release|published)\b",
        r"\b(find online|search for|look up online|google|web search)\b",
        r"\bhttps?://\b",
    ]
    for pattern in web_patterns:
        if re.search(pattern, q_lower):
            needs_web = True
            break

    # ── Step 5: Build reasoning string ───────────────────────
    reasoning_parts = []
    reasoning_parts.append(f"Intent: {best_intent.value} (score: {best_score:.2f})")
    reasoning_parts.append(f"Complexity: {complexity.value} (score: {complexity_score:.2f})")
    reasoning_parts.append(f"Word count: {word_count}")
    if is_followup:
        reasoning_parts.append("Detected as follow-up to conversation")
    if needs_web:
        reasoning_parts.append("Web search recommended")
    reasoning = " | ".join(reasoning_parts)

    return QueryClassification(
        intent=best_intent,
        complexity=complexity,
        confidence=best_score,
        recommended_model=recommended,
        prompt_template=best_intent.value,
        reasoning=reasoning,
        needs_web=needs_web,
        needs_vault=best_intent != QueryIntent.ACTION,
    )


def get_prompt_template(intent: QueryIntent) -> str:
    """Get the prompt template for a given intent."""
    return PROMPT_TEMPLATES.get(intent, PROMPT_TEMPLATES[QueryIntent.CHAT])


def build_prompt(classification: QueryClassification, query: str, context: str) -> str:
    """Build the final prompt using the classified intent's template."""
    template = get_prompt_template(classification.intent)
    return template.format(query=query, context=context)


# ── Available Models Helper ──────────────────────────────────

def get_available_models() -> list[str]:
    """Query Ollama for installed models."""
    try:
        import ollama
        models = ollama.list()
        return [m["name"].split(":")[0] for m in models.get("models", [])]
    except Exception:
        return []

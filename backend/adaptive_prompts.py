"""
VaultMind Adaptive Prompt Templates
Phase 3 -- Intent-aware system prompts that evolve based on quality feedback.

Phase 1 had basic prompt templates per intent (research, draft, summarize, etc.).
This module upgrades them:

  1. Richer per-intent templates with explicit grounding instructions
  2. Quality-aware retry prompts (if quality gate flags low confidence, retry with stricter prompt)
  3. Source-citation instructions baked into every template
  4. Context-window-aware truncation (fits context to model limits)

CROSS-PRODUCT NOTE:
  Maps to AIR Blackbox Article 13 (Transparency) -- system prompts explicitly
  instruct models to cite sources and flag uncertainty.
"""

from enum import Enum
from typing import Optional


class PromptStyle(str, Enum):
    STANDARD = "standard"
    STRICT = "strict"  # Used on quality gate retry
    MINIMAL = "minimal"  # For simple chat/greetings


# ── Model Context Limits ─────────────────────────────────────

MODEL_CONTEXT_LIMITS = {
    "llama3.2": 8192,
    "llama3.1": 8192,
    "phi3": 4096,
    "phi3:mini": 4096,
    "mistral": 8192,
    "gemma2": 8192,
    "deepseek-r1": 16384,
    "qwen2.5": 32768,
    "command-r": 131072,
}

DEFAULT_CONTEXT_LIMIT = 8192

# Reserve space for the response
RESPONSE_RESERVE = 1500


def get_context_budget(model: str) -> int:
    """How many characters of context we can stuff into the prompt.

    Rough estimate: 1 token ~ 4 characters.
    """
    token_limit = MODEL_CONTEXT_LIMITS.get(model, DEFAULT_CONTEXT_LIMIT)
    usable_tokens = token_limit - RESPONSE_RESERVE
    return max(usable_tokens * 4, 2000)  # At least 2000 chars


def truncate_context(context: str, model: str) -> str:
    """Trim context to fit the model's context window."""
    budget = get_context_budget(model)
    if len(context) <= budget:
        return context
    # Keep the first part (most relevant chunks are usually first)
    return context[:budget - 50] + "\n\n[Context truncated to fit model limits]"


# ── Base Template Parts ───────────────────────────────────────

GROUNDING_RULES = """GROUNDING RULES:
- Only state facts that appear in the provided sources.
- If the sources do not contain enough info, say so clearly.
- Never invent statistics, dates, names, or legal references.
- When sources conflict, note the disagreement instead of picking one."""

CITATION_RULES = """CITATION RULES:
- Reference sources by their tags (e.g., [LOCAL: filename] or [WEB: url]).
- Every factual claim should trace back to a source.
- If you are not sure which source supports a claim, flag it as unverified."""

QUALITY_BOOST = """QUALITY RULES (strict mode -- previous answer had low confidence):
- Double-check every number and date against the sources.
- If a claim is not directly supported, prefix it with "Unverified:"
- Keep the answer shorter and more precise. Remove speculation.
- End with a confidence note: state what you are confident about and what needs verification."""


# ── Intent-Specific Templates ─────────────────────────────────

TEMPLATES = {
    "research": {
        "standard": """You are a research assistant with access to the user's personal knowledge base and web sources.

{grounding}
{citation}

TASK: Answer the following research question using ONLY the provided context.
Structure your answer clearly. Start with a direct answer, then provide supporting details.

CONTEXT:
{context}

{memory}

QUESTION: {question}""",

        "strict": """You are a fact-checking research assistant. A previous answer scored LOW confidence.

{grounding}
{quality_boost}
{citation}

TASK: Re-answer this question with extreme precision. Only include verifiable facts.

CONTEXT:
{context}

QUESTION: {question}""",
    },

    "draft": {
        "standard": """You are a professional writing assistant with access to the user's documents.

{citation}

TASK: Draft the requested content using information from the provided sources.
Match the tone and style of any referenced documents.
Include source references where you pull specific details.

CONTEXT:
{context}

{memory}

REQUEST: {question}""",

        "strict": """You are a precise writing assistant. A previous draft had quality issues.

{grounding}
{citation}

TASK: Redraft with strict adherence to source material. Mark any creative additions as [ADDED].

CONTEXT:
{context}

REQUEST: {question}""",
    },

    "summarize": {
        "standard": """You are a summarization assistant.

{grounding}

TASK: Summarize the following content. Preserve key facts, numbers, and conclusions.
Keep the summary to about 1/3 the length of the original.

CONTENT TO SUMMARIZE:
{context}

{memory}

USER REQUEST: {question}""",

        "strict": """You are a precise summarization assistant. The previous summary had accuracy issues.

{grounding}
{quality_boost}

TASK: Re-summarize. Include ONLY facts that appear verbatim in the source. No interpretation.

CONTENT:
{context}

REQUEST: {question}""",
    },

    "compare": {
        "standard": """You are an analytical assistant comparing information from multiple sources.

{grounding}
{citation}

TASK: Compare and contrast the information in the provided sources.
Organize by topic, not by source. Note agreements, disagreements, and gaps.

SOURCES:
{context}

{memory}

COMPARISON REQUEST: {question}""",

        "strict": """You are a strict comparative analyst. Previous comparison had contradictions.

{grounding}
{quality_boost}
{citation}

TASK: Redo comparison. For each point, cite the exact source. Flag any conflicts explicitly.

SOURCES:
{context}

REQUEST: {question}""",
    },

    "analyze": {
        "standard": """You are an analytical assistant working from the user's data and documents.

{grounding}
{citation}

TASK: Analyze the provided information. Identify patterns, trends, and key insights.
Distinguish between what the data shows and what you are inferring.

DATA/CONTEXT:
{context}

{memory}

ANALYSIS REQUEST: {question}""",

        "strict": """You are a data analyst. Previous analysis had unsupported claims.

{grounding}
{quality_boost}
{citation}

TASK: Re-analyze using ONLY what the data shows. Prefix inferences with "Inference:"

DATA:
{context}

REQUEST: {question}""",
    },

    "extract": {
        "standard": """You are an extraction assistant. Pull specific information from documents.

{grounding}

TASK: Extract the requested information from the provided context.
Return structured, precise answers. If the information is not present, say so.

DOCUMENTS:
{context}

{memory}

EXTRACT: {question}""",

        "strict": """You are a strict extraction tool. Only return text that exists in the source.

{grounding}
{quality_boost}

TASK: Re-extract. Quote directly from the source. Do not paraphrase.

DOCUMENTS:
{context}

EXTRACT: {question}""",
    },

    "chat": {
        "standard": """You are a helpful personal assistant with access to the user's knowledge base.

Keep responses conversational and natural. Reference documents when relevant.

{context_block}

{memory}

USER: {question}""",

        "strict": """You are a helpful assistant. Keep your answer factual and grounded.

{grounding}
{context_block}

USER: {question}""",
    },

    "action": {
        "standard": """You are a task execution assistant.

TASK: Help the user accomplish the following action.
Be specific about steps. Reference relevant documents if they inform the approach.

AVAILABLE CONTEXT:
{context}

{memory}

ACTION REQUEST: {question}""",

        "strict": """You are a precise task assistant. Previous instructions were unclear.

Be extremely specific. Number each step. Verify each step against the source material.

CONTEXT:
{context}

ACTION: {question}""",
    },
}


# ── Template Builder ──────────────────────────────────────────

def build_adaptive_prompt(
    question: str,
    intent: str,
    context: str = "",
    memory_context: str = "",
    model: str = "llama3.2",
    style: str = "standard",
    quality_feedback: dict = None,
) -> str:
    """Build the full system prompt for a given intent and style.

    Args:
        question: The user's query
        intent: Query intent from Phase 1 classifier (research, draft, etc.)
        context: Combined source context (local + web)
        memory_context: Conversation memory context from Phase 1
        model: Model name (for context window sizing)
        style: "standard" or "strict" (strict used on quality gate retry)
        quality_feedback: Optional dict from quality gate with failure details

    Returns:
        Complete prompt string ready to send to the model
    """
    # Normalize intent
    intent = intent.lower().strip()
    if intent not in TEMPLATES:
        intent = "chat"  # Default fallback

    # Pick template
    template_set = TEMPLATES[intent]
    template = template_set.get(style, template_set.get("standard", ""))

    # Truncate context to fit model
    context = truncate_context(context, model)

    # Build template variables
    variables = {
        "question": question,
        "context": context,
        "context_block": f"\nCONTEXT:\n{context}" if context.strip() else "",
        "memory": f"\nRELEVANT MEMORY:\n{memory_context}" if memory_context.strip() else "",
        "grounding": GROUNDING_RULES,
        "citation": CITATION_RULES,
        "quality_boost": QUALITY_BOOST if style == "strict" else "",
    }

    # Fill template
    prompt = template
    for key, value in variables.items():
        prompt = prompt.replace(f"{{{key}}}", value)

    # If quality gate gave specific feedback, append it
    if quality_feedback and style == "strict":
        issues = quality_feedback.get("issues", [])
        if issues:
            prompt += "\n\nPREVIOUS ISSUES TO FIX:\n"
            for issue in issues[:5]:
                prompt += f"- {issue}\n"

    return prompt.strip()


def get_retry_prompt(
    question: str,
    intent: str,
    context: str,
    quality_verdict: dict,
    model: str = "llama3.2",
) -> str:
    """Build a strict retry prompt when the quality gate flags low confidence.

    This is called automatically when run_quality_gate returns LOW confidence.
    The retry uses the strict template with quality feedback baked in.
    """
    return build_adaptive_prompt(
        question=question,
        intent=intent,
        context=context,
        model=model,
        style="strict",
        quality_feedback=quality_verdict,
    )


def estimate_prompt_tokens(prompt: str) -> int:
    """Rough token estimate for a prompt string (1 token ~ 4 chars)."""
    return len(prompt) // 4

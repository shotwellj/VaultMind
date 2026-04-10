"""
VaultMind Context Fusion Engine
Phase 2 -- merges local vault retrieval with web search into a unified context.

The key insight: private context (client names, case details) is ONLY applied
AFTER fusion, inside the local environment. The web search only ever saw
the sanitized query. The LLM sees everything because it runs locally.

Architecture:
  [LOCAL chunks from ChromaDB]  ──┐
                                   ├──> Context Fusion ──> LLM (local Ollama)
  [WEB results from Search Proxy] ┘

Every piece of context is tagged with its source:
  [LOCAL: contract_v3.pdf, Section: Liability] ...chunk text...
  [WEB: nist.gov, Tier: 1] ...snippet text...

This lets the LLM (and the user) know exactly where each piece of info
came from, enabling proper citations.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FusedContext:
    """The unified context ready for the LLM."""
    context_text: str  # The merged context string
    local_count: int = 0  # Number of local chunks included
    web_count: int = 0  # Number of web results included
    sources: list = field(default_factory=list)  # All source labels
    source_tags: list = field(default_factory=list)  # [LOCAL] or [WEB] per source
    privacy_note: str = ""  # Note about what was stripped
    total_tokens_est: int = 0  # Rough token estimate


def fuse_contexts(
    local_chunks: list[tuple] = None,
    web_results: list = None,
    firewall_result=None,
    max_local_chunks: int = 8,
    max_web_results: int = 5,
    max_context_words: int = 3000,
) -> FusedContext:
    """Merge local vault chunks and web search results into unified context.

    Args:
        local_chunks: List of (doc_text, metadata_dict) from ChromaDB
        web_results: List of SearchResult objects from search_proxy
        firewall_result: FirewallResult showing what was stripped
        max_local_chunks: Max local chunks to include
        max_web_results: Max web results to include
        max_context_words: Rough word limit for total context

    Returns:
        FusedContext ready to inject into the system prompt
    """
    if local_chunks is None:
        local_chunks = []
    if web_results is None:
        web_results = []

    sections = []
    sources = []
    source_tags = []
    word_count = 0

    # ── Local context first (highest priority -- it's the user's own data) ──

    if local_chunks:
        local_parts = []
        for doc, meta in local_chunks[:max_local_chunks]:
            if word_count > max_context_words:
                break

            src = meta.get("source", "unknown") if isinstance(meta, dict) else "unknown"
            section = meta.get("section", "") if isinstance(meta, dict) else ""

            tag = f"[LOCAL: {src}"
            if section:
                tag += f", Section: {section}"
            tag += "]"

            local_parts.append(f"{tag}\n{doc}")
            sources.append(src)
            source_tags.append("LOCAL")
            word_count += len(doc.split())

        if local_parts:
            sections.append(
                "FROM YOUR PRIVATE DOCUMENTS (trusted, local-only):\n\n"
                + "\n\n---\n\n".join(local_parts)
            )

    # ── Web context second (lower priority, external) ──────────

    if web_results:
        web_parts = []
        for result in web_results[:max_web_results]:
            if word_count > max_context_words:
                break

            # Support both SearchResult objects and dicts
            title = result.title if hasattr(result, "title") else result.get("title", "")
            url = result.url if hasattr(result, "url") else result.get("url", "")
            snippet = result.snippet if hasattr(result, "snippet") else result.get("snippet", "")
            tier = result.quality_tier if hasattr(result, "quality_tier") else result.get("quality_tier", "")
            score = result.trust_score if hasattr(result, "trust_score") else result.get("trust_score", 0)

            tier_label = f", Tier: {tier.replace('tier', '')}" if tier else ""
            tag = f"[WEB: {url}{tier_label}]"

            entry = f"{tag}\nTitle: {title}\n{snippet}"
            web_parts.append(entry)
            sources.append(url)
            source_tags.append("WEB")
            word_count += len(snippet.split()) + len(title.split())

        if web_parts:
            sections.append(
                "FROM THE WEB (external, verify independently):\n\n"
                + "\n\n---\n\n".join(web_parts)
            )

    # ── Privacy note ──────────────────────────────────────────

    privacy_note = ""
    if firewall_result and firewall_result.was_modified:
        count = firewall_result.entity_count
        types = set()
        for e in firewall_result.entities_found:
            t = e.entity_type
            if hasattr(t, "value"):
                t = t.value
            types.add(t)
        type_str = ", ".join(sorted(types))
        privacy_note = (
            f"PRIVACY NOTE: {count} private entities ({type_str}) were detected in your query "
            f"and stripped before searching the web. Your full context is applied locally only."
        )

    # ── Build final context ───────────────────────────────────

    context_text = ""
    if privacy_note:
        context_text += privacy_note + "\n\n"
    context_text += "\n\n========\n\n".join(sections)

    return FusedContext(
        context_text=context_text,
        local_count=min(len(local_chunks), max_local_chunks),
        web_count=min(len(web_results), max_web_results),
        sources=sources,
        source_tags=source_tags,
        privacy_note=privacy_note,
        total_tokens_est=word_count * 4 // 3,  # Rough: 1 word ~ 1.3 tokens
    )


def build_fusion_prompt(
    fused: FusedContext,
    intent: str = "research",
) -> str:
    """Build a system prompt that leverages fused context properly.

    Args:
        fused: FusedContext from fuse_contexts()
        intent: Query intent from Query Intelligence Engine

    Returns:
        System prompt string ready for Ollama
    """
    has_local = fused.local_count > 0
    has_web = fused.web_count > 0

    if has_local and has_web:
        mode_instruction = (
            "You have access to BOTH the user's private documents AND web search results.\n"
            "Private documents are MORE trustworthy than web results.\n"
            "When information conflicts, prefer the private documents and note the discrepancy.\n"
            "Tag every claim with its source: [LOCAL] or [WEB].\n"
        )
    elif has_local:
        mode_instruction = (
            "You have access ONLY to the user's private documents.\n"
            "ONLY use information present in these documents.\n"
            "If the answer is not in the documents, say so clearly.\n"
        )
    elif has_web:
        mode_instruction = (
            "You have access to web search results.\n"
            "ONLY use information from the provided results.\n"
            "NEVER fabricate URLs or facts not in the results.\n"
            "Cite sources with their URLs.\n"
        )
    else:
        mode_instruction = (
            "No relevant information was found in documents or web search.\n"
            "Let the user know and suggest how to improve their query.\n"
        )

    # Intent-specific additions
    intent_additions = {
        "research": "Provide thorough analysis with citations for every claim.",
        "summarize": "Be concise. Preserve key facts and their sources.",
        "compare": "Present information in a structured comparison format.",
        "draft": "Use the context as reference material for the writing task.",
        "analyze": "Provide deep analysis, noting patterns and implications.",
        "extract": "Pull out the specific information requested, with exact sources.",
    }
    intent_line = intent_additions.get(intent, "Be clear, concise, and cite your sources.")

    prompt = (
        f"You are VaultMind, a privacy-first AI assistant.\n\n"
        f"RULES:\n"
        f"1. NEVER invent information not present in the sources below.\n"
        f"2. NEVER fabricate or guess URLs.\n"
        f"3. {intent_line}\n"
        f"4. {mode_instruction}\n"
        f"SOURCES:\n{fused.context_text}"
    )

    return prompt

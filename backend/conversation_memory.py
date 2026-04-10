"""
VaultMind Conversation Memory Layer
Stores conversation summaries in ChromaDB so the AI can recall past discussions.

How it works:
  1. After each conversation, a summary is generated and embedded
  2. When a new query comes in, relevant past conversations are retrieved
  3. This gives VaultMind "long-term memory" across sessions

Storage: Uses a separate ChromaDB collection ("conversation_memory") to avoid
polluting the document vault.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

# ChromaDB and Ollama are imported at call time to avoid circular imports

MEMORY_COLLECTION = "conversation_memory"
EMBED_MODEL = "nomic-embed-text"


def get_memory_collection(chroma_client):
    """Get or create the conversation memory collection."""
    return chroma_client.get_or_create_collection(
        name=MEMORY_COLLECTION,
        metadata={"hnsw:space": "l2"}
    )


def summarize_conversation(messages: list[dict], model: str = "mistral") -> str:
    """Generate a short summary of a conversation using the local LLM.

    Args:
        messages: List of {role, content} message dicts
        model: Ollama model to use for summarization

    Returns:
        A 2-3 sentence summary of what was discussed
    """
    import ollama

    if not messages or len(messages) < 2:
        return ""

    # Build a compact version of the conversation (last 10 messages max)
    convo_text = ""
    for msg in messages[-10:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            convo_text += f"{role}: {content}\n"

    if not convo_text.strip():
        return ""

    try:
        result = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize this conversation in 2-3 sentences. "
                        "Focus on: what the user asked about, what topics were discussed, "
                        "and any key conclusions or decisions. Be factual and concise."
                    )
                },
                {"role": "user", "content": convo_text}
            ],
            options={"temperature": 0}
        )
        return result["message"]["content"].strip()
    except Exception as e:
        print(f"[ConversationMemory] Summary generation failed: {e}")
        return ""


def store_conversation_memory(
    chroma_client,
    conversation_id: str,
    summary: str,
    messages: list[dict],
    model: str = "mistral",
):
    """Store a conversation summary in the memory collection.

    Args:
        chroma_client: ChromaDB client instance
        conversation_id: Unique ID for this conversation
        summary: Pre-generated summary (or empty to auto-generate)
        messages: The conversation messages
        model: Ollama model name for embedding + summarization
    """
    import ollama

    if not summary:
        summary = summarize_conversation(messages, model)
    if not summary:
        return  # Nothing to store

    col = get_memory_collection(chroma_client)

    # Generate embedding for the summary
    try:
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=summary)["embedding"]
    except Exception as e:
        print(f"[ConversationMemory] Embedding failed: {e}")
        return

    # Extract key topics from the conversation for metadata
    user_messages = [m["content"] for m in messages if m.get("role") == "user" and m.get("content")]
    topics = " | ".join(user_messages[:5])  # First 5 user messages as topic hint
    if len(topics) > 300:
        topics = topics[:300]

    metadata = {
        "conversation_id": conversation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message_count": len(messages),
        "topics": topics,
        "type": "conversation_summary",
    }

    col.upsert(
        ids=[f"memory_{conversation_id}"],
        embeddings=[embedding],
        documents=[summary],
        metadatas=[metadata],
    )
    print(f"[ConversationMemory] Stored memory for conversation {conversation_id}")


def recall_relevant_memories(
    chroma_client,
    query: str,
    n_results: int = 3,
    max_age_days: Optional[int] = None,
) -> list[dict]:
    """Retrieve past conversation summaries relevant to the current query.

    Args:
        chroma_client: ChromaDB client instance
        query: The current user query
        n_results: Max number of memories to retrieve
        max_age_days: Only return memories from the last N days (None = all)

    Returns:
        List of dicts with keys: summary, conversation_id, timestamp, topics
    """
    import ollama

    col = get_memory_collection(chroma_client)

    # Check if there are any memories stored
    try:
        count = col.count()
        if count == 0:
            return []
    except Exception:
        return []

    try:
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    except Exception:
        return []

    try:
        results = col.query(
            query_embeddings=[embedding],
            n_results=min(n_results, count),
        )
    except Exception:
        return []

    if not results["documents"] or not results["documents"][0]:
        return []

    memories = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Only include reasonably relevant memories (L2 distance threshold)
        if dist > 1.2:
            continue

        # Optional age filter
        if max_age_days and meta.get("timestamp"):
            try:
                stored_at = datetime.fromisoformat(meta["timestamp"])
                age = (datetime.now(timezone.utc) - stored_at).days
                if age > max_age_days:
                    continue
            except (ValueError, TypeError):
                pass

        memories.append({
            "summary": doc,
            "conversation_id": meta.get("conversation_id", ""),
            "timestamp": meta.get("timestamp", ""),
            "topics": meta.get("topics", ""),
            "relevance": round(1.0 - (dist / 2.0), 3),  # Normalize to 0-1 score
        })

    return memories


def build_memory_context(memories: list[dict]) -> str:
    """Format retrieved memories into a context block for the system prompt."""
    if not memories:
        return ""

    parts = ["RELEVANT PAST CONVERSATIONS:"]
    for i, mem in enumerate(memories, 1):
        ts = mem.get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                ts = dt.strftime("%b %d, %Y")
            except (ValueError, TypeError):
                pass
        parts.append(f"\n{i}. [{ts}] {mem['summary']}")

    return "\n".join(parts)

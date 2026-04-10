"""
VaultMind Knowledge Graph
Phase 4 -- Relationship maps between entities in the knowledge base.

When you ask about Entity A, the graph automatically surfaces related
entities (documents, people, topics, dates) that you might not have
thought to search for. This is the "I forgot we had that document" killer.

Architecture:
  - NetworkX directed graph stored as JSON
  - Nodes: documents, people, topics, dates, organizations
  - Edges: mentions, references, co-occurs, authored_by, dated
  - Entity extraction via regex + optional spaCy NER
  - Auto-rebuilds when new documents are indexed

Storage: ~/.vaultmind/graph/knowledge_graph.json
No cloud. Everything local.
"""

import os
import re
import json
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

try:
    import networkx as nx
    NX_OK = True
except ImportError:
    NX_OK = False
    print("[KnowledgeGraph] networkx not installed. Run: pip install networkx")


# ── Config ────────────────────────────────────────────────────

GRAPH_DIR = os.path.expanduser("~/.vaultmind/graph")
GRAPH_PATH = os.path.join(GRAPH_DIR, "knowledge_graph.json")


# ── Node Types ────────────────────────────────────────────────

class NodeType:
    DOCUMENT = "document"
    PERSON = "person"
    ORGANIZATION = "organization"
    TOPIC = "topic"
    DATE = "date"
    LOCATION = "location"
    LEGAL_REF = "legal_reference"  # Article X, Section Y, Statute Z


class EdgeType:
    MENTIONS = "mentions"  # Document mentions a person/org/topic
    REFERENCES = "references"  # Document references another document
    CO_OCCURS = "co_occurs"  # Two entities appear in the same chunk
    AUTHORED_BY = "authored_by"  # Document authored by person
    DATED = "dated"  # Entity associated with a date
    RELATED_TO = "related_to"  # General relationship


# ── Entity Extraction ─────────────────────────────────────────

# Patterns for extracting entities from text
ENTITY_PATTERNS = {
    NodeType.PERSON: [
        # Titled names: Mr. Smith, Dr. Johnson, Prof. Williams
        re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Judge|Hon|Sen|Rep)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'),
        # Two+ word capitalized names (after punctuation + space)
        re.compile(r'(?<=\.\s\s)[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'),
    ],
    NodeType.ORGANIZATION: [
        # Common org patterns
        re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Co|Association|Institute|University|Commission|Agency|Department|Board|Council|Foundation)\b'),
        # Acronym orgs (3+ uppercase letters)
        re.compile(r'\b[A-Z]{3,6}\b(?=\s+(?:said|announced|reported|released|published|requires|mandates))'),
    ],
    NodeType.DATE: [
        # Full dates: January 15, 2024 or 15 January 2024
        re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'),
        re.compile(r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'),
        # ISO dates
        re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
        # Quarter references
        re.compile(r'\bQ[1-4]\s+\d{4}\b'),
    ],
    NodeType.LEGAL_REF: [
        re.compile(r'\bArticle\s+\d+(?:\(\d+\))?'),
        re.compile(r'\bSection\s+\d+(?:\.\d+)*'),
        re.compile(r'\bClause\s+\d+(?:\.\d+)*'),
        re.compile(r'\b\d+\s+(?:U\.?S\.?C\.?|CFR|USC)\s+[^\s,]+'),
    ],
    NodeType.TOPIC: [
        # These are identified by frequency, not pattern.
        # Placeholder: we'll extract topics from high-frequency noun phrases later.
    ],
}

# Common words to skip when extracting entities
SKIP_WORDS = {
    "the", "this", "that", "these", "those", "their", "there",
    "here", "where", "when", "what", "which", "who", "how",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october",
    "november", "december",
}


def extract_entities(text: str, source: str = "") -> list:
    """Extract named entities from text using regex patterns.

    Returns a list of dicts: [{"text": "...", "type": "person", "source": "file.pdf"}]
    """
    entities = []
    seen = set()

    for node_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                entity_text = match.group().strip()
                # Skip short/common words
                if len(entity_text) < 3:
                    continue
                if entity_text.lower() in SKIP_WORDS:
                    continue
                # Deduplicate
                key = (entity_text.lower(), node_type)
                if key in seen:
                    continue
                seen.add(key)
                entities.append({
                    "text": entity_text,
                    "type": node_type,
                    "source": source,
                })

    # Extract high-frequency topic-like terms (nouns > 6 chars appearing 3+ times)
    words = re.findall(r'\b[A-Za-z]{6,}\b', text)
    word_freq = defaultdict(int)
    for w in words:
        word_freq[w.lower()] += 1
    for word, count in word_freq.items():
        if count >= 3 and word not in SKIP_WORDS:
            key = (word, NodeType.TOPIC)
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": word.title(),
                    "type": NodeType.TOPIC,
                    "source": source,
                })

    return entities


def _entity_id(entity_text: str, entity_type: str) -> str:
    """Create a unique node ID for an entity."""
    clean = entity_text.strip().lower()
    return f"{entity_type}:{hashlib.md5(clean.encode()).hexdigest()[:10]}"


# ── Graph Operations ──────────────────────────────────────────

def create_graph():
    """Create a new empty knowledge graph."""
    if not NX_OK:
        return None
    return nx.DiGraph()


def load_graph() -> Optional[object]:
    """Load the knowledge graph from disk."""
    if not NX_OK:
        return None

    if os.path.exists(GRAPH_PATH):
        try:
            with open(GRAPH_PATH) as f:
                data = json.load(f)
            G = nx.node_link_graph(data)
            return G
        except Exception as e:
            print(f"[KnowledgeGraph] Failed to load graph: {e}")

    return create_graph()


def save_graph(G) -> bool:
    """Save the knowledge graph to disk."""
    if not NX_OK or G is None:
        return False

    os.makedirs(GRAPH_DIR, exist_ok=True)
    try:
        data = nx.node_link_data(G)
        with open(GRAPH_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"[KnowledgeGraph] Failed to save graph: {e}")
        return False


def add_document_to_graph(G, source: str, text: str, metadata: dict = None):
    """Extract entities from a document and add them to the graph.

    Args:
        G: NetworkX graph
        source: Document filename/path
        text: Document text content
        metadata: Optional dict with section_header, indexed_at, etc.
    """
    if G is None:
        return

    metadata = metadata or {}
    doc_id = _entity_id(source, NodeType.DOCUMENT)

    # Add the document node
    G.add_node(doc_id, **{
        "label": source,
        "type": NodeType.DOCUMENT,
        "indexed_at": metadata.get("indexed_at", datetime.utcnow().isoformat()),
        "section": metadata.get("section_header", ""),
    })

    # Extract entities from the text
    entities = extract_entities(text, source)

    for entity in entities:
        eid = _entity_id(entity["text"], entity["type"])

        # Add entity node (or update if exists)
        if not G.has_node(eid):
            G.add_node(eid, **{
                "label": entity["text"],
                "type": entity["type"],
                "first_seen": datetime.utcnow().isoformat(),
                "mention_count": 0,
            })

        # Increment mention count
        G.nodes[eid]["mention_count"] = G.nodes[eid].get("mention_count", 0) + 1

        # Add edge: document -> entity (mentions)
        if G.has_edge(doc_id, eid):
            G[doc_id][eid]["weight"] = G[doc_id][eid].get("weight", 0) + 1
        else:
            G.add_edge(doc_id, eid, type=EdgeType.MENTIONS, weight=1)

    # Add co-occurrence edges between entities found in the same text
    entity_ids = [_entity_id(e["text"], e["type"]) for e in entities]
    for i, eid_a in enumerate(entity_ids):
        for eid_b in entity_ids[i + 1:]:
            if eid_a != eid_b:
                if G.has_edge(eid_a, eid_b):
                    G[eid_a][eid_b]["weight"] = G[eid_a][eid_b].get("weight", 0) + 1
                else:
                    G.add_edge(eid_a, eid_b, type=EdgeType.CO_OCCURS, weight=1)

    return G


def rebuild_graph(documents: list) -> object:
    """Rebuild the entire graph from a list of documents.

    Args:
        documents: List of dicts with keys: source, text, metadata
    """
    G = create_graph()
    if G is None:
        return None

    for doc in documents:
        add_document_to_graph(
            G,
            source=doc.get("source", "unknown"),
            text=doc.get("text", ""),
            metadata=doc.get("metadata", {}),
        )

    save_graph(G)
    return G


# ── Query Functions ───────────────────────────────────────────

def find_related(G, query: str, max_hops: int = 2, max_results: int = 10) -> list:
    """Find entities related to a query by searching the graph.

    Searches for nodes whose labels match query keywords,
    then traverses up to max_hops edges to find related entities.

    Returns a list of dicts: [{"entity": "...", "type": "...", "relevance": 0.9, "path": [...]}]
    """
    if G is None or not NX_OK:
        return []

    query_words = set(query.lower().split())

    # Find starting nodes that match query keywords
    start_nodes = []
    for node_id, data in G.nodes(data=True):
        label = data.get("label", "").lower()
        label_words = set(label.split())
        overlap = query_words & label_words
        if overlap:
            score = len(overlap) / max(len(query_words), 1)
            start_nodes.append((node_id, score))

    if not start_nodes:
        return []

    # Sort by match score
    start_nodes.sort(key=lambda x: x[1], reverse=True)

    # BFS from top starting nodes
    related = {}
    for start_id, start_score in start_nodes[:5]:
        # Get neighbors up to max_hops away
        visited = {start_id}
        frontier = [(start_id, 0, [start_id])]

        while frontier:
            current, depth, path = frontier.pop(0)
            if depth >= max_hops:
                continue

            for neighbor in G.neighbors(current):
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                edge_data = G[current][neighbor]
                edge_weight = edge_data.get("weight", 1)
                # Relevance decays with hops, increases with edge weight
                relevance = start_score * (0.7 ** depth) * min(edge_weight / 5, 1.0)

                new_path = path + [neighbor]
                if neighbor not in related or related[neighbor]["relevance"] < relevance:
                    node_data = G.nodes[neighbor]
                    related[neighbor] = {
                        "entity": node_data.get("label", neighbor),
                        "type": node_data.get("type", "unknown"),
                        "relevance": relevance,
                        "mention_count": node_data.get("mention_count", 0),
                        "path": [G.nodes[n].get("label", n) for n in new_path],
                        "edge_type": edge_data.get("type", "related"),
                    }

                frontier.append((neighbor, depth + 1, new_path))

    # Sort by relevance and return top results
    results = sorted(related.values(), key=lambda x: x["relevance"], reverse=True)
    return results[:max_results]


def get_document_connections(G, source: str) -> dict:
    """Get all entities connected to a specific document.

    Returns a dict grouped by entity type.
    """
    if G is None:
        return {}

    doc_id = _entity_id(source, NodeType.DOCUMENT)
    if not G.has_node(doc_id):
        return {}

    connections = defaultdict(list)
    for neighbor in G.neighbors(doc_id):
        data = G.nodes[neighbor]
        entity_type = data.get("type", "unknown")
        connections[entity_type].append({
            "entity": data.get("label", neighbor),
            "mention_count": data.get("mention_count", 0),
            "edge_weight": G[doc_id][neighbor].get("weight", 1),
        })

    # Sort each group by mention count
    for entity_type in connections:
        connections[entity_type].sort(key=lambda x: x["mention_count"], reverse=True)

    return dict(connections)


def get_graph_stats(G) -> dict:
    """Get summary stats about the knowledge graph."""
    if G is None:
        return {"status": "graph not initialized"}

    node_types = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_types[data.get("type", "unknown")] += 1

    edge_types = defaultdict(int)
    for _, _, data in G.edges(data=True):
        edge_types[data.get("type", "unknown")] += 1

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
        "density": nx.density(G) if G.number_of_nodes() > 1 else 0,
    }


def build_context_from_graph(G, query: str, max_entities: int = 5) -> str:
    """Build additional context from the knowledge graph for a query.

    This is injected into the system prompt alongside vault/web context
    to surface related entities the user might not have searched for.
    """
    related = find_related(G, query, max_hops=2, max_results=max_entities)
    if not related:
        return ""

    lines = ["RELATED ENTITIES FROM YOUR KNOWLEDGE BASE:"]
    for r in related:
        entity = r["entity"]
        etype = r["type"]
        mentions = r.get("mention_count", 0)
        path = " > ".join(r.get("path", []))
        lines.append(f"  - {entity} ({etype}, {mentions} mentions) via: {path}")

    return "\n".join(lines)

"""VaultMind MCP server.

Gives Claude Desktop, Claude Code, Cursor, or any other MCP client access to
a local VaultMind vault — without uploading the documents anywhere.

Be precise about what that means. Your corpus stays on your machine. The
passages returned by these tools are sent by the client to whatever model it
uses, which for a cloud client means they leave. The difference from
uploading your files is that you choose what leaves, one question at a time,
and VaultMind records every passage it hands over — visible in the privacy
panel and at /mcp/disclosures.

This process is deliberately a thin shim. It does not touch ChromaDB or
Ollama directly; it calls the running VaultMind backend, so retrieval logic
lives in one place and nothing can be retrieved without being logged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 60.0

mcp = FastMCP("vaultmind")


def _base_url() -> str:
    return os.environ.get("VAULTMIND_URL", DEFAULT_BASE_URL).rstrip("/")


def _candidate_token_paths() -> list[Path]:
    """Where the backend may have written its access token.

    Mirrors the backend's DATA_DIR resolution: an explicit override, the
    Electron app's data directory, then a source checkout.
    """
    paths = []
    if env := os.environ.get("VAULTMIND_DATA_DIR"):
        paths.append(Path(env) / ".vaultmind_token")
    paths.append(
        Path.home() / "Library" / "Application Support" / "VaultMind"
        / "data" / ".vaultmind_token"
    )
    paths.append(Path(__file__).resolve().parents[2] / "backend" / ".vaultmind_token")
    return paths


def _token() -> str:
    if env := os.environ.get("VAULTMIND_TOKEN"):
        return env
    for path in _candidate_token_paths():
        try:
            token = path.read_text().strip()
            if token:
                return token
        except OSError:
            continue
    return ""


class VaultUnavailable(RuntimeError):
    """Raised with guidance the user can act on, rather than a stack trace."""


def _call(method: str, path: str, **kwargs) -> dict:
    token = _token()
    if not token:
        raise VaultUnavailable(
            "Could not find the VaultMind access token. Start VaultMind "
            "(`bash start.sh`), or set VAULTMIND_TOKEN in the MCP server's "
            "environment. Looked in: "
            + ", ".join(str(p) for p in _candidate_token_paths())
        )
    try:
        response = httpx.request(
            method,
            f"{_base_url()}{path}",
            headers={"X-VaultMind-Token": token},
            timeout=TIMEOUT,
            **kwargs,
        )
    except httpx.ConnectError:
        raise VaultUnavailable(
            f"VaultMind is not running at {_base_url()}. Start it with "
            "`bash start.sh`, or set VAULTMIND_URL if it listens elsewhere."
        )
    except httpx.TimeoutException:
        raise VaultUnavailable(
            "VaultMind did not respond in time. A first query can be slow "
            "while the embedding model loads — try again."
        )

    if response.status_code == 401:
        raise VaultUnavailable(
            "VaultMind rejected the access token. It is regenerated if the "
            "data directory is cleared — restart your MCP client to pick up "
            "the current one."
        )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def vault_search(query: str, limit: int = 5) -> str:
    """Search the user's private local document vault.

    Use this whenever the user asks about their own documents, contracts,
    notes, email, records, or anything that sounds personal and would not be
    answerable from general knowledge — "what does my lease say", "when did I
    sign that", "what were the payment terms we agreed".

    Returns excerpts with the source each came from. Cite the sources in your
    answer. If nothing comes back, say so rather than guessing — an empty
    result means the vault has nothing relevant, not that the answer is
    unknown.

    Args:
        query: A natural-language question or topic.
        limit: Maximum passages to return (1-10, default 5).
    """
    data = _call("POST", "/mcp/search", json={"query": query, "limit": limit})

    if data.get("error"):
        return f"VaultMind could not search: {data['error']}"

    results = data.get("results") or []
    if not results:
        return (
            f"No relevant passages found in the vault for {query!r}. "
            "The vault may not contain anything on this topic."
        )

    lines = [f"{len(results)} passage(s) from the user's vault:\n"]
    for i, r in enumerate(results, 1):
        header = f"[{i}] {r['source']}"
        if r.get("section"):
            header += f" — {r['section']}"
        lines.append(f"{header}\n{r['excerpt']}\n")
    lines.append(
        "These passages came from the user's private local vault. "
        "Cite the sources above by name."
    )
    return "\n".join(lines)


@mcp.tool()
def vault_list_sources() -> str:
    """List the documents indexed in the user's vault.

    Names and sizes only — no content. Useful for answering "what do you have
    access to", or to find the exact source name before calling
    vault_get_document.
    """
    data = _call("GET", "/mcp/sources")
    sources = data.get("sources") or []
    if not sources:
        return "The vault is empty — no documents are indexed yet."

    lines = [f"{data.get('total', len(sources))} source(s) in the vault:\n"]
    lines += [f"  - {s['source']} ({s['chunks']} chunks)" for s in sources[:200]]
    if len(sources) > 200:
        lines.append(f"  … and {len(sources) - 200} more")
    return "\n".join(lines)


@mcp.tool()
def vault_get_document(source: str, max_chars: int = 6000) -> str:
    """Retrieve more of one specific document from the vault.

    Use only after vault_search has identified a document and the excerpt is
    not enough — for example when the user asks you to summarise a whole
    contract. Prefer vault_search: it is cheaper and returns less of the
    user's private material.

    Args:
        source: Exact source name, as given by vault_search or
            vault_list_sources.
        max_chars: Maximum characters to return (500-20000, default 6000).
    """
    data = _call("POST", "/mcp/document",
                 json={"source": source, "max_chars": max_chars})

    if data.get("error"):
        return (
            f"{data['error']} Call vault_list_sources to see the exact names "
            "of indexed documents."
        )

    text = data.get("text", "")
    suffix = ""
    if data.get("truncated"):
        suffix = (
            f"\n\n[Truncated at {max_chars} characters. "
            "Use vault_search to find specific passages instead.]"
        )
    return f"{source}:\n\n{text}{suffix}"


def main() -> None:
    try:
        mcp.run()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()

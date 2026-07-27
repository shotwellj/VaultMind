"""Tests for the MCP surface.

Deliberately avoid anything needing Ollama or a populated vault so these
run in CI. The parts worth protecting are the contract and the disclosure
logging, not the retrieval quality — that is covered elsewhere.
"""

import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.join(ROOT, "mcp"))

os.environ.setdefault("VAULTMIND_DATA_DIR", tempfile.mkdtemp())


# ── Backend contract ───────────────────────────────────────────

def test_mcp_routes_exist():
    import main

    paths = {getattr(r, "path", None) for r in main.app.routes}
    for path in ("/mcp/search", "/mcp/sources", "/mcp/document",
                 "/mcp/disclosures"):
        assert path in paths


def test_mcp_routes_require_a_token():
    """These return vault content — they must not be publicly reachable."""
    import main

    for path in ("/mcp/search", "/mcp/sources", "/mcp/document",
                 "/mcp/disclosures"):
        assert path not in main.PUBLIC_PATHS


def test_excerpt_and_result_caps_are_bounded():
    """The amount returned is the amount that reaches a cloud model."""
    import main

    assert 0 < main.MCP_EXCERPT_CHARS <= 4000
    assert 0 < main.MCP_MAX_RESULTS <= 25


# ── Disclosure logging ─────────────────────────────────────────

def test_disclosure_is_recorded_with_query_and_sources():
    import egress_log

    before = egress_log.disclosure_summary()["retrievals"]
    egress_log.disclose(
        tool="vault_search",
        query="what are the payment terms",
        sources=["contract.pdf", "amendment.pdf"],
        characters=842,
    )
    summary = egress_log.disclosure_summary()

    assert summary["retrievals"] == before + 1
    assert "contract.pdf" in summary["sources_touched"]

    latest = egress_log.recent_disclosures(1)[0]
    assert latest["tool"] == "vault_search"
    assert latest["query"] == "what are the payment terms"
    assert latest["characters"] == 842


def test_disclosures_appear_in_the_privacy_snapshot():
    """The point of the log is that it shows up where users look."""
    import egress_log

    egress_log.disclose("vault_search", "q", ["a.pdf"], 10)
    assert "disclosures" in egress_log.snapshot()


def test_disclosure_query_is_truncated():
    """A pathological query should not be stored unbounded."""
    import egress_log

    egress_log.disclose("vault_search", "x" * 5000, ["a.pdf"], 10)
    assert len(egress_log.recent_disclosures(1)[0]["query"]) <= 300


# ── MCP server ─────────────────────────────────────────────────

def test_server_exposes_exactly_the_intended_tools():
    import asyncio

    from vaultmind_mcp import server

    tools = asyncio.run(server.mcp.list_tools())
    assert {t.name for t in tools} == {
        "vault_search", "vault_list_sources", "vault_get_document",
    }


def test_tools_describe_when_to_use_them():
    """The description is the whole interface — a client picks tools from it."""
    import asyncio

    from vaultmind_mcp import server

    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert "vault" in (tools["vault_search"].description or "").lower()
    # get_document is the expensive one; it should point at search instead.
    assert "vault_search" in (tools["vault_get_document"].description or "")


def test_missing_token_produces_actionable_guidance(monkeypatch):
    """"Not running" is the most common failure — it must not be a traceback."""
    from vaultmind_mcp import server

    monkeypatch.setenv("VAULTMIND_DATA_DIR", tempfile.mkdtemp())
    monkeypatch.delenv("VAULTMIND_TOKEN", raising=False)
    monkeypatch.setattr(server, "_token", lambda: "")

    with pytest.raises(server.VaultUnavailable) as excinfo:
        server._call("GET", "/mcp/sources")

    message = str(excinfo.value)
    assert "start.sh" in message or "VAULTMIND_TOKEN" in message


def test_backend_down_is_reported_not_raised_as_connection_error(monkeypatch):
    from vaultmind_mcp import server

    monkeypatch.setattr(server, "_token", lambda: "fake-token")
    # Port 9 is discard — reliably refuses.
    monkeypatch.setenv("VAULTMIND_URL", "http://127.0.0.1:9")

    with pytest.raises(server.VaultUnavailable) as excinfo:
        server._call("GET", "/mcp/sources")

    assert "not running" in str(excinfo.value).lower()

"""Regression tests for bugs that shipped in v1.0.0.

Each test here maps to something that was broken in a released build and
was not caught because nothing ever ran the backend. They are deliberately
cheap: no Ollama, no network, no model downloads, so they can run in CI.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

os.environ.setdefault("VAULTMIND_DATA_DIR", tempfile.mkdtemp())


# ── Routing ────────────────────────────────────────────────────

def test_no_duplicate_routes():
    """A second @app.post("/agent") shadowed the first and 422'd the UI."""
    import main

    seen, dupes = set(), []
    for route in main.app.routes:
        for method in getattr(route, "methods", []) or []:
            key = (method, getattr(route, "path", None))
            if key in seen:
                dupes.append(key)
            seen.add(key)
    assert not dupes, f"shadowed routes: {dupes}"


def test_agent_route_accepts_the_payload_the_ui_sends():
    """The UI posts {message: ...}; the LAM schema wanted {query: ...}."""
    import main

    import inspect

    agent = next(
        r for r in main.app.routes
        if getattr(r, "path", "") == "/agent" and "POST" in (r.methods or [])
    )
    body_types = [
        p.annotation for p in
        inspect.signature(agent.endpoint).parameters.values()
        if inspect.isclass(p.annotation)
    ]
    assert main.ChatMessage in body_types, (
        f"/agent must accept the chat payload shape, got {body_types}"
    )


# ── Retrieval ──────────────────────────────────────────────────

def test_vault_uses_cosine_distance():
    """Default L2 put distances in the hundreds, so no chunk ever passed."""
    import main

    assert main.VAULT_SPACE.get("hnsw:space") == "cosine"


def test_thresholds_are_in_cosine_range():
    """Guards against a threshold tuned for one metric being left on another."""
    import main

    assert 0 < main.RELEVANCE_THRESHOLD < 2
    assert main.RELEVANCE_THRESHOLD <= main.RELEVANCE_THRESHOLD_FALLBACK < 2


# ── SSRF ───────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000/",
    "http://localhost:9999/",
    "http://169.254.169.254/latest/meta-data/",
    "http://192.168.1.1/",
    "http://10.0.0.5/admin",
    "http://[::1]:8000/",
    "http://0.0.0.0:8000/",
    "file:///etc/passwd",
    "ftp://example.com/x",
    "not-a-url",
])
def test_internal_and_non_http_urls_are_refused(url):
    from net_guard import BlockedURL, assert_fetchable_url

    with pytest.raises(BlockedURL):
        assert_fetchable_url(url)


def test_public_urls_are_allowed():
    from net_guard import assert_fetchable_url

    assert assert_fetchable_url("https://example.com")


# ── Agent tool confinement ─────────────────────────────────────

@pytest.mark.parametrize("tool,params", [
    ("summarize_document", {"file_path": "/etc/passwd"}),
    ("extract_dates", {"file_path": "/etc/hosts"}),
    ("tag_document", {"file_path": "/etc/hosts", "tags": ["x"]}),
    ("move_file_internal", {"source": "/etc/hosts", "destination": "/tmp/x"}),
    ("create_document",
     {"matter_id": "../../../../tmp/evil", "doc_type": "d", "content": "c"}),
    ("log_time_entry",
     {"matter_id": "../../etc", "hours": 1, "description": "d"}),
])
def test_agent_tools_cannot_escape_allowed_roots(tool, params):
    import lam

    result = str(lam.execute_tool(tool, params))
    assert "outside the allowed" in result or "Unsafe path" in result, result


def test_destructive_tools_are_not_auto_executed():
    """move_file_internal used to run with no human approval."""
    import lam

    assert "move_file_internal" not in lam.AUTO_EXECUTE_TOOLS
    assert "move_file_internal" in lam.STAGED_TOOLS
    assert not (lam.AUTO_EXECUTE_TOOLS & lam.STAGED_TOOLS)


# ── Planner robustness ─────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    '{"reasoning":"r","steps":[]}',
    '```json\n{"reasoning":"r","steps":[]}\n```',
    'Sure, here is the plan:\n{"reasoning":"r","steps":[]}\nHope that helps.',
    '{"reasoning":"braces {inside} a string","steps":[]}',
])
def test_planner_survives_model_chatter(raw):
    """A bare json.loads meant any prose around the JSON killed the request."""
    import lam

    assert isinstance(lam._parse_plan(raw), dict)


def test_planner_rejects_garbage():
    import lam

    with pytest.raises(Exception):
        lam._parse_plan("I'm sorry, I can't help with that.")


# ── Auth ───────────────────────────────────────────────────────

def test_cors_is_not_wide_open():
    """allow_origins=['*'] let any site read the vault cross-origin."""
    import main

    assert "*" not in main.ALLOWED_ORIGINS


def test_sensitive_paths_are_not_public():
    import main

    for path in ("/files", "/chat", "/query", "/debug-chunks", "/lam/agent"):
        assert path not in main.PUBLIC_PATHS


# ── Privacy panel ──────────────────────────────────────────────

def test_privacy_reports_observed_connections_not_a_hardcoded_list():
    """It used to return external_connections: [] no matter what happened."""
    import egress_log
    import socket

    egress_log.install()
    before = egress_log.snapshot()["external_connection_count"]

    # A real connection to a real external address, via CPython sockets.
    try:
        s = socket.create_connection(("example.com", 80), timeout=10)
        s.close()
    except OSError:
        pytest.skip("no network available")

    after = egress_log.snapshot()
    assert after["external_connection_count"] > before
    assert any("example.com" in h["host"] for h in after["external"])


def test_loopback_is_not_counted_as_external():
    """Ollama runs on localhost; calling it is not data leaving the machine."""
    import egress_log

    egress_log.install()
    egress_log._record("127.0.0.1", 11434)
    snap = egress_log.snapshot()
    assert any(h["host"] == "127.0.0.1" for h in snap["local"])
    assert not any(h["host"] == "127.0.0.1" for h in snap["external"])


def test_unobservable_requests_are_marked_declared_not_observed():
    """ddgs uses primp, a Rust client that bypasses CPython sockets.

    Reporting those as observed would make a self-report look like a
    measurement — the exact failure this module exists to avoid.
    """
    import egress_log

    egress_log.declare("duckduckgo.com", "web search")
    snap = egress_log.snapshot()

    entry = next(h for h in snap["external"] if h["host"] == "duckduckgo.com")
    assert entry["source"] == "declared"
    assert "duckduckgo.com" in snap["declared_hosts"]


def test_privacy_endpoint_exposes_no_static_assurance():
    """Guards against a hardcoded 'nothing ever leaves' string coming back."""
    import inspect

    import main

    source = inspect.getsource(main.privacy_dashboard)
    assert "never for your personal data" not in source
    assert '"external_connections": []' not in source

"""Tests for the MCP client (harness Phase 2).

No real MCP servers are spawned and no network is touched: registration,
risk tiers, result flattening, and the sync/async call bridge are all
exercised with fakes, matching the cheap-in-CI style of the other suites.
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ.setdefault("VAULTMIND_DATA_DIR", tempfile.mkdtemp())

import agent_loop  # noqa: E402
import mcp_client  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_client, "MCP_SERVERS_FILE",
                        str(tmp_path / "mcp_servers.json"))
    saved_tools = dict(agent_loop.TOOLS)
    yield
    agent_loop.TOOLS.clear()
    agent_loop.TOOLS.update(saved_tools)


def fake_tool(name, description="a tool", schema=None):
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object",
                               "properties": {"x": {"type": "string"}}},
    )


# ── Config ─────────────────────────────────────────────────────

def test_upsert_and_remove_roundtrip():
    mcp_client.upsert_server("files", {"command": "npx", "args": ["-y", "pkg"]})

    cfg = mcp_client.load_config()
    assert cfg["servers"]["files"]["command"] == "npx"
    assert cfg["servers"]["files"]["enabled"] is True
    assert cfg["servers"]["files"]["auto_tools"] == []

    assert mcp_client.remove_server("files") is True
    assert mcp_client.remove_server("files") is False
    assert mcp_client.load_config()["servers"] == {}


@pytest.mark.parametrize("name,spec", [
    ("bad name!", {"command": "npx"}),          # invalid name chars
    ("x" * 41, {"command": "npx"}),             # name too long
    ("ok", {"command": ""}),                    # empty command
    ("ok", {"command": "npx", "args": "-y"}),   # args not a list
    ("ok", {"command": "npx", "env": {"A": 1}}),  # env values not strings
    ("ok", {"command": "npx", "auto_tools": "read"}),  # auto_tools not a list
])
def test_invalid_specs_are_rejected(name, spec):
    with pytest.raises(ValueError):
        mcp_client.validate_spec(name, spec)


def test_corrupt_config_file_degrades_to_empty():
    with open(mcp_client.MCP_SERVERS_FILE, "w") as f:
        f.write("{not json")
    assert mcp_client.load_config() == {"servers": {}}


# ── Registration and risk tiers ────────────────────────────────

def test_external_tools_register_staged_by_default():
    registered = mcp_client.register_server_tools(
        "github", {"auto_tools": []},
        [fake_tool("create_issue"), fake_tool("get_repo")],
        call_fn=lambda *a: "ok",
    )

    assert registered == ["mcp_github_create_issue", "mcp_github_get_repo"]
    for name in registered:
        tool = agent_loop.TOOLS[name]
        assert tool.risk == "staged", (
            f"{name} must require approval — VaultMind cannot vet external tools"
        )
        assert "external MCP server 'github'" in tool.description


def test_auto_tools_allowlist_promotes_named_tools_only():
    mcp_client.register_server_tools(
        "fs", {"auto_tools": ["read_file"]},
        [fake_tool("read_file"), fake_tool("write_file")],
        call_fn=lambda *a: "ok",
    )

    assert agent_loop.TOOLS["mcp_fs_read_file"].risk == "auto"
    assert agent_loop.TOOLS["mcp_fs_write_file"].risk == "staged"


def test_registration_never_clobbers_an_existing_tool():
    agent_loop.register_tool(agent_loop.Tool(
        name="mcp_evil_vault_search", description="pre-existing",
        parameters={"type": "object", "properties": {}},
        risk="auto", handler=lambda **kw: "original",
    ))

    registered = mcp_client.register_server_tools(
        "evil", {}, [fake_tool("vault_search")], call_fn=lambda *a: "hijacked",
    )

    assert registered == []
    assert agent_loop.TOOLS["mcp_evil_vault_search"].handler() == "original"


def test_bad_input_schema_falls_back_to_empty_object():
    mcp_client.register_server_tools(
        "s", {}, [fake_tool("t", schema={"type": "array"})],
        call_fn=lambda *a: "ok",
    )
    assert agent_loop.TOOLS["mcp_s_t"].parameters == {
        "type": "object", "properties": {}
    }


def test_tool_names_are_sanitized():
    assert mcp_client.registered_name("my server", "do/thing") == "mcp_my_server_do_thing"


def test_handler_declares_egress_and_forwards_args(monkeypatch):
    declared = []
    monkeypatch.setattr(mcp_client.egress_log, "declare",
                        lambda host, purpose, port=443: declared.append(host))
    calls = []
    mcp_client.register_server_tools(
        "slack", {}, [fake_tool("post")],
        call_fn=lambda server, tool, args: calls.append((server, tool, args)) or "sent",
    )

    result = agent_loop.TOOLS["mcp_slack_post"].handler(channel="#general", text="hi")

    assert result == "sent"
    assert calls == [("slack", "post", {"channel": "#general", "text": "hi"})]
    assert declared == ["mcp:slack"], "every external call must be egress-declared"


# ── Result flattening ──────────────────────────────────────────

def test_result_text_joins_text_content():
    result = SimpleNamespace(content=[
        SimpleNamespace(type="text", text="line one"),
        SimpleNamespace(type="image", data="..."),
        SimpleNamespace(type="text", text="line two"),
    ], isError=False)
    assert mcp_client.result_text(result) == (
        "line one\n[image content omitted]\nline two"
    )


def test_result_text_marks_errors_and_empties():
    err = SimpleNamespace(content=[SimpleNamespace(type="text", text="boom")],
                          isError=True)
    assert mcp_client.result_text(err) == "Tool error: boom"
    assert mcp_client.result_text(SimpleNamespace(content=[], isError=False)) == \
        "(empty result)"


# ── The sync/async bridge ──────────────────────────────────────

def test_manager_call_bridges_into_the_loop_thread():
    mgr = mcp_client.MCPManager()

    class FakeSession:
        async def call_tool(self, tool, arguments):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text",
                                         text=f"{tool}:{json.dumps(arguments)}")],
                isError=False,
            )

    mgr.sessions["srv"] = FakeSession()
    try:
        assert mgr.call("srv", "echo", {"a": 1}) == 'echo:{"a": 1}'
    finally:
        mgr.shutdown()


def test_manager_call_reports_disconnected_and_failing_servers():
    mgr = mcp_client.MCPManager()

    class FailingSession:
        async def call_tool(self, tool, arguments):
            raise RuntimeError("pipe closed")

    mgr.sessions["down"] = FailingSession()
    try:
        assert "not connected" in mgr.call("ghost", "t", {})
        assert "pipe closed" in mgr.call("down", "t", {})
    finally:
        mgr.shutdown()


def test_connect_without_sdk_gives_install_hint(monkeypatch):
    monkeypatch.setattr(mcp_client, "MCP_AVAILABLE", False)
    mgr = mcp_client.MCPManager()
    with pytest.raises(RuntimeError, match="pip install"):
        mgr.connect("x", {"command": "npx"})


# ── Status / integration ───────────────────────────────────────

def test_status_reflects_config_and_connection_state():
    mcp_client.upsert_server("files", {"command": "npx", "args": ["-y", "pkg"],
                                       "auto_tools": ["read_file"]})
    mgr = mcp_client.MCPManager()
    mgr.errors["files"] = "spawn failed"

    status = mgr.status()

    assert isinstance(status["mcp_available"], bool)
    entry = status["servers"]["files"]
    assert entry["connected"] is False
    assert entry["error"] == "spawn failed"
    assert entry["auto_tools"] == ["read_file"]


def test_main_exposes_mcp_management_routes():
    import main

    paths = {getattr(r, "path", "") for r in main.app.routes}
    for path in ("/agent/mcp", "/agent/mcp/servers", "/agent/mcp/servers/{name}"):
        assert path in paths, f"missing route {path}"

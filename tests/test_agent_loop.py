"""Tests for the agent loop (Phase 1 of the harness).

Deliberately cheap, like test_regressions.py: no Ollama, no network. The
model round is scripted by replacing agent_loop._chat, and all on-disk
state (runs, staged queue, audit) is redirected to a tmp dir.
"""

import json
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ.setdefault("VAULTMIND_DATA_DIR", tempfile.mkdtemp())

import agent_loop  # noqa: E402
import lam  # noqa: E402


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Keep every test's runs, staged queue, and audit out of the repo."""
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    monkeypatch.setattr(agent_loop, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(lam, "STAGED_QUEUE_FILE", str(tmp_path / "staged.json"))
    monkeypatch.setattr(lam, "AUDIT_DIR", str(audit_dir))

    saved_tools = dict(agent_loop.TOOLS)
    yield
    agent_loop.TOOLS.clear()
    agent_loop.TOOLS.update(saved_tools)


def scripted_chat(monkeypatch, responses):
    """Make _chat return the given responses in order; fail if exhausted."""
    it = iter(responses)
    calls = []

    def _fake_chat(model, messages, tools):
        calls.append({"model": model, "messages": list(messages), "tools": tools})
        try:
            return next(it)
        except StopIteration:
            raise AssertionError("model was called more times than scripted")

    monkeypatch.setattr(agent_loop, "_chat", _fake_chat)
    return calls


def assistant(content="", tool_calls=None):
    msg = {"content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {"function": {"name": name, "arguments": args}}
            for name, args in tool_calls
        ]
    return {"message": msg}


def register(name, risk, handler, params=None, required=None):
    agent_loop.register_tool(agent_loop.Tool(
        name=name,
        description=f"test tool {name}",
        parameters={"type": "object",
                    "properties": params or {},
                    "required": required or []},
        risk=risk,
        handler=handler,
    ))


# ── Termination ────────────────────────────────────────────────

def test_answer_without_tools_finishes_in_one_step(monkeypatch):
    scripted_chat(monkeypatch, [assistant("The answer is 4.")])

    run = agent_loop.start_run("what is 2+2?")

    assert run["status"] == "done"
    assert run["answer"] == "The answer is 4."
    assert run["steps_used"] == 1
    assert not run["hit_step_limit"]


def test_step_limit_terminates_a_loop_that_never_finishes(monkeypatch):
    register("noop", "auto", lambda **kw: "ok")
    scripted_chat(monkeypatch, [
        assistant("", [("noop", {})]) for _ in range(3)
    ])

    run = agent_loop.start_run("loop forever", max_steps=3)

    assert run["status"] == "done"
    assert run["hit_step_limit"]
    assert run["steps_used"] == 3
    assert "step limit" in run["answer"]


def test_empty_goal_is_rejected():
    with pytest.raises(ValueError):
        agent_loop.start_run("   ")


def test_model_failure_marks_run_error(monkeypatch):
    def _boom(model, messages, tools):
        raise RuntimeError("ollama is down")
    monkeypatch.setattr(agent_loop, "_chat", _boom)

    run = agent_loop.start_run("anything")

    assert run["status"] == "error"
    assert "ollama is down" in run["error"]


# ── Tool dispatch ──────────────────────────────────────────────

def test_auto_tool_executes_and_result_feeds_back(monkeypatch):
    seen = []
    register("lookup", "auto", lambda **kw: seen.append(kw) or "result-42",
             params={"q": {"type": "string"}}, required=["q"])
    scripted_chat(monkeypatch, [
        assistant("", [("lookup", {"q": "meaning"})]),
        assistant("It is 42."),
    ])

    run = agent_loop.start_run("look it up")

    assert run["status"] == "done"
    assert seen == [{"q": "meaning"}]
    tool_msgs = [m for m in run["messages"] if m["role"] == "tool"]
    assert tool_msgs == [{"role": "tool", "tool_name": "lookup", "content": "result-42"}]
    assert run["steps"][0]["tool"] == "lookup"
    # every executed tool must land in the shared audit trail
    audit_files = os.listdir(lam.AUDIT_DIR)
    assert len(audit_files) == 1
    with open(os.path.join(lam.AUDIT_DIR, audit_files[0])) as f:
        assert json.load(f)["tool"] == "lookup"


def test_unknown_tool_does_not_kill_the_run(monkeypatch):
    scripted_chat(monkeypatch, [
        assistant("", [("made_up_tool", {})]),
        assistant("done without it"),
    ])

    run = agent_loop.start_run("try a fake tool")

    assert run["status"] == "done"
    tool_msgs = [m for m in run["messages"] if m["role"] == "tool"]
    assert "Unknown tool" in tool_msgs[0]["content"]


def test_matter_id_is_injected_when_model_forgets_it(monkeypatch):
    seen = {}
    register("file_it", "auto", lambda **kw: seen.update(kw) or "filed",
             params={"matter_id": {"type": "string"}, "what": {"type": "string"}},
             required=["matter_id", "what"])
    scripted_chat(monkeypatch, [
        assistant("", [("file_it", {"what": "notes"})]),
        assistant("filed"),
    ])

    run = agent_loop.start_run("file my notes", matter_id="case-7")

    assert run["status"] == "done"
    assert seen == {"what": "notes", "matter_id": "case-7"}


def test_tool_results_are_truncated(monkeypatch):
    register("huge", "auto", lambda **kw: "x" * 100_000)
    scripted_chat(monkeypatch, [
        assistant("", [("huge", {})]),
        assistant("ok"),
    ])

    run = agent_loop.start_run("fetch something huge")

    tool_msg = next(m for m in run["messages"] if m["role"] == "tool")
    assert len(tool_msg["content"]) == agent_loop.TOOL_RESULT_MAX_CHARS


# ── Pause / resume ─────────────────────────────────────────────

def _paused_run(monkeypatch, handler_calls):
    register("send_report", "staged",
             lambda **kw: handler_calls.append(kw) or "sent",
             params={"to": {"type": "string"}}, required=["to"])
    scripted_chat(monkeypatch, [
        assistant("", [("send_report", {"to": "a@b.c"})]),
    ])
    return agent_loop.start_run("send the report")


def test_staged_tool_pauses_the_run_without_executing(monkeypatch):
    calls = []
    run = _paused_run(monkeypatch, calls)

    assert run["status"] == "paused"
    assert calls == [], "staged tool must not execute before approval"
    pending = run["pending_action"]
    assert pending["tool"] == "send_report"
    assert pending["params"] == {"to": "a@b.c"}

    staged = lam.load_staged()
    assert len(staged) == 1
    assert staged[0]["run_id"] == run["id"]
    assert staged[0]["status"] == "pending"


def test_approval_executes_and_resumes(monkeypatch):
    calls = []
    run = _paused_run(monkeypatch, calls)

    scripted_chat(monkeypatch, [assistant("Report sent.")])
    resumed = agent_loop.resume_run(run["id"], approved=True)

    assert calls == [{"to": "a@b.c"}], "approval must execute exactly once"
    assert resumed["status"] == "done"
    assert resumed["answer"] == "Report sent."
    assert resumed["pending_action"] is None
    tool_msg = next(m for m in resumed["messages"] if m["role"] == "tool")
    assert tool_msg["content"] == "sent"
    assert lam.load_staged()[0]["status"] == "approved"


def test_rejection_does_not_execute_and_the_model_replans(monkeypatch):
    calls = []
    run = _paused_run(monkeypatch, calls)

    scripted_chat(monkeypatch, [assistant("Understood, I won't send it.")])
    resumed = agent_loop.resume_run(run["id"], approved=False)

    assert calls == [], "rejected action must never execute"
    assert resumed["status"] == "done"
    tool_msg = next(m for m in resumed["messages"] if m["role"] == "tool")
    assert "REJECTED" in tool_msg["content"]
    assert lam.load_staged()[0]["status"] == "rejected"


def test_resume_requires_a_paused_run(monkeypatch):
    scripted_chat(monkeypatch, [assistant("done")])
    run = agent_loop.start_run("quick answer")

    result = agent_loop.resume_run(run["id"], approved=True)
    assert "error" in result

    assert "error" in agent_loop.resume_run("no-such-run", approved=True)


def test_tool_calls_after_a_staged_pause_are_reported_skipped(monkeypatch):
    executed = []
    register("send_report", "staged", lambda **kw: "sent")
    register("after", "auto", lambda **kw: executed.append(1) or "ran")
    scripted_chat(monkeypatch, [
        assistant("", [("send_report", {}), ("after", {})]),
    ])
    run = agent_loop.start_run("send then do more")
    assert run["status"] == "paused"
    assert executed == [], "calls after the pause point must not run"

    scripted_chat(monkeypatch, [assistant("done")])
    resumed = agent_loop.resume_run(run["id"], approved=True)

    skipped = [m for m in resumed["messages"]
               if m["role"] == "tool" and m["tool_name"] == "after"]
    assert skipped and "Not executed" in skipped[0]["content"]


# ── Persistence ────────────────────────────────────────────────

def test_runs_persist_and_reload(monkeypatch):
    scripted_chat(monkeypatch, [assistant("saved answer")])
    run = agent_loop.start_run("persist me")

    reloaded = agent_loop.load_run(run["id"])
    assert reloaded == run

    summaries = agent_loop.list_runs()
    assert summaries[0]["id"] == run["id"]
    assert "messages" not in summaries[0], "summaries must not dump transcripts"


# ── Registry / integration ─────────────────────────────────────

def test_lam_tools_inherit_lams_risk_tiers():
    """The loop and the one-shot planner must agree on what needs approval."""
    for name in ("tag_document", "summarize_document", "extract_dates",
                 "log_time_entry"):
        assert agent_loop.TOOLS[name].risk == "auto", name
    for name in ("create_document", "draft_email", "create_calendar_event",
                 "move_file_internal", "check_conflicts"):
        assert agent_loop.TOOLS[name].risk == "staged", name


def test_registry_rejects_unknown_risk_tier():
    with pytest.raises(ValueError):
        register("bad", "yolo", lambda **kw: None)


def test_main_registers_retrieval_tools_and_run_routes():
    import main

    for name in ("vault_search", "web_search", "fetch_url"):
        assert name in agent_loop.TOOLS, f"{name} not registered by main.py"
        assert agent_loop.TOOLS[name].risk == "auto"

    paths = {getattr(r, "path", "") for r in main.app.routes}
    for path in ("/agent/run", "/agent/runs", "/agent/runs/{run_id}",
                 "/agent/runs/{run_id}/approve", "/agent/runs/{run_id}/reject"):
        assert path in paths, f"missing route {path}"

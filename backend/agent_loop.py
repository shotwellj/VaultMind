"""
VaultMind Agent Loop — Phase 1 of the agent harness.

Replaces plan-then-execute with a real observe-act loop: the model calls
tools through Ollama's native tool-calling API, sees each result, and
decides the next step. Runs persist to disk so a STAGED tool can pause the
run for human approval and resume with the outcome instead of being
fire-and-forget.

Safety model is inherited from lam.py, not reinvented:
  - AUTO tools execute inside the loop; STAGED tools pause it.
  - Every executed tool writes the same audit record (lam.write_audit),
    so /audit-log stays the single trail.
  - Staged actions land in the same queue file as before, tagged with a
    run_id so approval resumes the run.
  - File tools keep lam's path confinement; network tools are registered
    by main.py on top of its SSRF guard and egress log.

A run is synchronous within one request: /agent/run returns when the run
finishes, errors, or pauses for approval. Streaming is a later phase.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import lam

# ── Storage ────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(
    os.environ.get("VAULTMIND_DATA_DIR", BASE_DIR), "agent_runs"
)

DEFAULT_MAX_STEPS = 10
MAX_STEPS_CAP = 25
TOOL_RESULT_MAX_CHARS = 4000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_path(run_id: str) -> str:
    return os.path.join(RUNS_DIR, f"{run_id}.json")


def save_run(run: dict) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    run["updated"] = _now()
    with open(_run_path(run["id"]), "w") as f:
        json.dump(run, f, indent=2)


def load_run(run_id: str) -> Optional[dict]:
    try:
        with open(_run_path(run_id)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def list_runs(limit: int = 50) -> list[dict]:
    """Newest-first run summaries (no message transcripts)."""
    if not os.path.isdir(RUNS_DIR):
        return []
    paths = sorted(
        (os.path.join(RUNS_DIR, n) for n in os.listdir(RUNS_DIR) if n.endswith(".json")),
        key=os.path.getmtime,
        reverse=True,
    )[:limit]
    summaries = []
    for p in paths:
        try:
            with open(p) as f:
                run = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        summaries.append(run_summary(run))
    return summaries


def run_summary(run: dict) -> dict:
    return {
        "id": run["id"],
        "goal": run["goal"],
        "status": run["status"],
        "steps_used": run["steps_used"],
        "max_steps": run["max_steps"],
        "pending_action": run.get("pending_action"),
        "answer": run.get("answer"),
        "error": run.get("error"),
        "created": run.get("created"),
        "updated": run.get("updated"),
    }


# ── Tool Registry ──────────────────────────────────────────────

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict            # JSON schema for the arguments object
    risk: str                   # "auto" executes in-loop, "staged" pauses
    handler: Callable[..., Any] = field(repr=False, default=None)


TOOLS: dict[str, Tool] = {}


def register_tool(tool: Tool) -> None:
    if tool.risk not in ("auto", "staged"):
        raise ValueError(f"Unknown risk tier {tool.risk!r} for {tool.name}")
    TOOLS[tool.name] = tool


def unregister_tool(name: str) -> None:
    TOOLS.pop(name, None)


def _ollama_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in TOOLS.values()
    ]


def _str_param(desc: str) -> dict:
    return {"type": "string", "description": desc}


def _register_lam_tools() -> None:
    """Expose lam.py's implementations as loop tools.

    Risk tiers come from lam's AUTO_EXECUTE_TOOLS / STAGED_TOOLS sets so the
    two entry points (one-shot planner and this loop) can never disagree
    about what needs approval.
    """
    schemas: dict[str, tuple[str, dict, list[str]]] = {
        "create_document": (
            "Create a new document in a matter folder (prep notes, timeline, summary).",
            {
                "matter_id": _str_param("Matter/case folder the document belongs to."),
                "doc_type": _str_param("Kind of document, e.g. 'timeline', 'summary'."),
                "content": _str_param("Full markdown body of the document."),
                "title": _str_param("Optional human-readable title."),
            },
            ["matter_id", "doc_type", "content"],
        ),
        "move_file_internal": (
            "Move a file between allowed folders. Destructive — requires approval.",
            {
                "source": _str_param("Path of the file to move."),
                "destination": _str_param("Path to move it to."),
            },
            ["source", "destination"],
        ),
        "tag_document": (
            "Add metadata tags to a document for filtering and search.",
            {
                "file_path": _str_param("Path of the document to tag."),
                "tags": {"type": "array", "items": {"type": "string"},
                         "description": "Tags to add."},
            },
            ["file_path", "tags"],
        ),
        "summarize_document": (
            "Summarize one document file in a few sentences.",
            {"file_path": _str_param("Path of the document to summarize.")},
            ["file_path"],
        ),
        "extract_dates": (
            "Extract all dates and deadlines from a document file.",
            {"file_path": _str_param("Path of the document to scan.")},
            ["file_path"],
        ),
        "log_time_entry": (
            "Log billable time to a matter.",
            {
                "matter_id": _str_param("Matter to log time against."),
                "hours": {"type": "number", "description": "Hours worked."},
                "description": _str_param("What the time was spent on."),
            },
            ["matter_id", "hours", "description"],
        ),
        "check_conflicts": (
            "Check whether party names appear in existing matters (conflict of interest).",
            {
                "party_names": {"type": "array", "items": {"type": "string"},
                                "description": "Names to check."},
            },
            ["party_names"],
        ),
        "draft_email": (
            "Stage an email draft for review. NEVER auto-sends.",
            {
                "to": _str_param("Recipient address."),
                "subject": _str_param("Subject line."),
                "body": _str_param("Email body."),
            },
            ["to", "subject", "body"],
        ),
        "create_calendar_event": (
            "Create a calendar event or deadline reminder.",
            {
                "title": _str_param("Event title."),
                "date": _str_param("Event date, ISO format preferred."),
                "duration": _str_param("Duration, e.g. '1h'."),
            },
            ["title", "date"],
        ),
    }
    for name, (description, props, required) in schemas.items():
        register_tool(Tool(
            name=name,
            description=description,
            parameters={"type": "object", "properties": props, "required": required},
            risk="auto" if name in lam.AUTO_EXECUTE_TOOLS else "staged",
            handler=(lambda _name: lambda **params: lam.execute_tool(_name, params))(name),
        ))


_register_lam_tools()


# ── Ollama plumbing ────────────────────────────────────────────

def _chat(model: str, messages: list[dict], tools: list[dict]) -> Any:
    """One model round. Isolated so tests can replace it without Ollama."""
    import ollama
    return ollama.chat(
        model=model,
        messages=messages,
        tools=tools,
        options={"temperature": 0.2},
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or an ollama response object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_call(call: Any) -> tuple[str, dict]:
    fn = _get(call, "function", {})
    name = str(_get(fn, "name", "") or "")
    args = _get(fn, "arguments", {}) or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


SYSTEM_PROMPT = """You are VaultMind's local agent. You accomplish the user's goal by calling tools, one step at a time, reacting to each result.

Rules:
- Use vault_search before answering anything about the user's own documents.
- AUTO tools run immediately. STAGED tools pause the run until the user approves — only call one when the goal genuinely requires it.
- If the user rejects an action, do not retry it; adjust the plan or finish.
- When the goal is accomplished (or clearly impossible), reply with your final answer as plain text and make NO tool calls."""


# ── The Loop ───────────────────────────────────────────────────

def start_run(
    goal: str,
    matter_id: str = "",
    model: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict:
    """Create a run and drive it until it finishes, errors, or pauses."""
    goal = (goal or "").strip()
    if not goal:
        raise ValueError("Empty goal")

    run = {
        "id": str(uuid.uuid4()),
        "created": _now(),
        "goal": goal,
        "matter_id": matter_id,
        "model": model or os.environ.get("VAULTMIND_SLM_MODEL", "qwen2.5"),
        "max_steps": max(1, min(int(max_steps or DEFAULT_MAX_STEPS), MAX_STEPS_CAP)),
        "steps_used": 0,
        "status": "running",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ],
        "steps": [],
        "answer": None,
        "error": None,
        "hit_step_limit": False,
        "pending_action": None,
    }
    save_run(run)
    return _loop(run)


def _loop(run: dict) -> dict:
    while run["steps_used"] < run["max_steps"]:
        try:
            resp = _chat(run["model"], run["messages"], _ollama_tools())
        except Exception as e:
            run["status"] = "error"
            run["error"] = f"Model call failed: {e}"
            save_run(run)
            return run

        msg = _get(resp, "message", {})
        content = _get(msg, "content", "") or ""
        tool_calls = _get(msg, "tool_calls", None) or []
        run["steps_used"] += 1

        assistant: dict = {"role": "assistant", "content": content}
        if tool_calls:
            assistant["tool_calls"] = [
                {"function": {"name": n, "arguments": a}}
                for n, a in (_parse_call(c) for c in tool_calls)
            ]
        run["messages"].append(assistant)

        if not tool_calls:
            run["status"] = "done"
            run["answer"] = content
            save_run(run)
            return run

        for i, call in enumerate(tool_calls):
            name, args = _parse_call(call)
            tool = TOOLS.get(name)

            if tool is None:
                available = ", ".join(sorted(TOOLS))
                _append_tool_result(
                    run, name, f"Unknown tool: {name!r}. Available tools: {available}"
                )
                continue

            if tool.risk == "staged":
                remaining = [
                    _parse_call(c)[0] for c in tool_calls[i + 1:]
                ]
                action_id = _queue_staged(run, name, args)
                run["pending_action"] = {
                    "action_id": action_id,
                    "tool": name,
                    "params": args,
                    "remaining_calls": remaining,
                }
                run["status"] = "paused"
                save_run(run)
                return run

            _execute(run, tool, args)

        save_run(run)

    run["status"] = "done"
    run["hit_step_limit"] = True
    run["answer"] = run.get("answer") or (
        "Stopped: reached the step limit before finishing. "
        "Review the steps taken so far and start a new run to continue."
    )
    save_run(run)
    return run


def _execute(run: dict, tool: Tool, args: dict) -> None:
    """Run an AUTO (or approved) tool, audit it, feed the result back."""
    # Convenience: matter-scoped tools often need the run's matter_id and
    # small local models routinely forget to pass it.
    if (
        run.get("matter_id")
        and "matter_id" in tool.parameters.get("properties", {})
        and not args.get("matter_id")
    ):
        args = {**args, "matter_id": run["matter_id"]}

    try:
        result = tool.handler(**args)
    except TypeError as e:
        result = f"Tool error: bad arguments for {tool.name}: {e}"
    except Exception as e:
        result = f"Tool error: {e}"

    audit_id = lam.write_audit(
        tool.name, args, result,
        reasoning=f"agent loop step {run['steps_used']} of run {run['id']}",
        auto_executed=True,
        matter_id=run.get("matter_id", ""),
    )
    text = _append_tool_result(run, tool.name, str(result))
    run["steps"].append({
        "step": run["steps_used"],
        "tool": tool.name,
        "params": args,
        "result": text,
        "audit_id": audit_id,
    })


def _append_tool_result(run: dict, tool_name: str, result: str) -> str:
    text = (result or "")[:TOOL_RESULT_MAX_CHARS]
    run["messages"].append({"role": "tool", "tool_name": tool_name, "content": text})
    return text


# ── Staged actions: pause and resume ───────────────────────────

def _queue_staged(run: dict, tool: str, params: dict) -> str:
    """Queue a staged action in lam's queue, tagged with this run's id."""
    action_id = str(uuid.uuid4())
    actions = lam.load_staged()
    actions.append({
        "id": action_id,
        "timestamp": _now(),
        "tool": tool,
        "params": params,
        "reasoning": f"agent loop step {run['steps_used']} of run {run['id']}",
        "matter_id": run.get("matter_id", ""),
        "status": "pending",
        "run_id": run["id"],
    })
    lam.save_staged(actions)
    lam.write_audit(
        tool, params, "STAGED — awaiting approval (run paused)",
        reasoning=f"agent loop step {run['steps_used']} of run {run['id']}",
        auto_executed=False,
        matter_id=run.get("matter_id", ""),
    )
    return action_id


def _mark_staged(action_id: str, status: str) -> None:
    actions = lam.load_staged()
    for action in actions:
        if action["id"] == action_id and action["status"] == "pending":
            action["status"] = status
            action[f"{status}_at"] = _now()
    lam.save_staged(actions)


def resume_run(run_id: str, approved: bool) -> dict:
    """Resolve a paused run's pending action and continue the loop.

    Approving executes the staged tool and hands the result back to the
    model; rejecting hands back a refusal so the model can re-plan or
    finish. Either way the run keeps its momentum instead of dying in the
    queue.
    """
    run = load_run(run_id)
    if run is None:
        return {"error": f"Run not found: {run_id}"}
    if run["status"] != "paused" or not run.get("pending_action"):
        return {"error": f"Run {run_id} is not awaiting approval "
                         f"(status: {run['status']})."}

    pending = run["pending_action"]
    run["pending_action"] = None
    _mark_staged(pending["action_id"], "approved" if approved else "rejected")

    tool = TOOLS.get(pending["tool"])
    if approved:
        if tool is None:
            _append_tool_result(run, pending["tool"],
                                f"Tool error: {pending['tool']} is no longer registered.")
        else:
            _execute(run, tool, pending["params"])
    else:
        text = _append_tool_result(
            run, pending["tool"],
            "The user REJECTED this action. Do not retry it; "
            "adjust the plan or finish.",
        )
        run["steps"].append({
            "step": run["steps_used"],
            "tool": pending["tool"],
            "params": pending["params"],
            "result": text,
            "rejected": True,
        })

    for name in pending.get("remaining_calls", []):
        _append_tool_result(
            run, name,
            "Not executed — a prior action in this step required approval. "
            "Call this tool again if it is still needed.",
        )

    run["status"] = "running"
    save_run(run)
    return _loop(run)

"""
VaultMind MCP client — Phase 2 of the agent harness.

VaultMind already *serves* MCP (other assistants query the vault). This
module is the other direction: VaultMind *consumes* MCP servers, so the
agent loop can use any tool the ecosystem ships — filesystem, GitHub,
Slack — while keeping VaultMind's governance wrapped around every call.

Trust model:
  - Every external tool registers as STAGED. VaultMind cannot vet what a
    third-party server does with its arguments, so a human approves each
    call by default. A user can promote specific tools to auto via the
    per-server `auto_tools` allowlist — an explicit, named opt-in.
  - Every call is declared in the egress log before it runs: the
    arguments were chosen by the model and are leaving VaultMind's
    process for code it does not control.
  - Executed calls land in the same lam audit trail as every other tool,
    because they run through the agent loop like any registered tool.

Plumbing note: the MCP SDK is asyncio; the agent loop is synchronous.
A single dedicated event-loop thread owns every server connection, and
each connection lives inside its own task (the SDK's stdio contexts must
enter and exit in the same task). Tool handlers bridge in with
run_coroutine_threadsafe.

The `mcp` package is an optional dependency — without it, config
endpoints still work but connections report how to enable support.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from typing import Any, Optional

import agent_loop
import egress_log

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# ── Config ─────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_SERVERS_FILE = os.path.join(
    os.environ.get("VAULTMIND_DATA_DIR", BASE_DIR), "mcp_servers.json"
)

CONNECT_TIMEOUT = 30.0
CALL_TIMEOUT = 60.0
TOOL_PREFIX = "mcp"

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def load_config() -> dict:
    try:
        with open(MCP_SERVERS_FILE) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"servers": {}}
    if not isinstance(cfg, dict) or not isinstance(cfg.get("servers"), dict):
        return {"servers": {}}
    return cfg


def save_config(cfg: dict) -> None:
    os.makedirs(os.path.dirname(MCP_SERVERS_FILE), exist_ok=True)
    with open(MCP_SERVERS_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def validate_spec(name: str, spec: dict) -> dict:
    """Normalize a server spec, rejecting anything malformed loudly."""
    if not _NAME_RE.match(name or ""):
        raise ValueError(
            f"Invalid server name {name!r}: use 1-40 letters, digits, _ or -."
        )
    command = str(spec.get("command", "")).strip()
    if not command:
        raise ValueError(f"Server {name!r} needs a non-empty 'command'.")
    args = spec.get("args", [])
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ValueError(f"Server {name!r}: 'args' must be a list of strings.")
    env = spec.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise ValueError(f"Server {name!r}: 'env' must map strings to strings.")
    auto_tools = spec.get("auto_tools", [])
    if not isinstance(auto_tools, list) or not all(
        isinstance(t, str) for t in auto_tools
    ):
        raise ValueError(f"Server {name!r}: 'auto_tools' must be a list of tool names.")
    return {
        "command": command,
        "args": args,
        "env": env,
        "enabled": bool(spec.get("enabled", True)),
        "auto_tools": auto_tools,
    }


def upsert_server(name: str, spec: dict) -> dict:
    normalized = validate_spec(name, spec)
    cfg = load_config()
    cfg["servers"][name] = normalized
    save_config(cfg)
    return normalized


def remove_server(name: str) -> bool:
    cfg = load_config()
    existed = name in cfg["servers"]
    cfg["servers"].pop(name, None)
    save_config(cfg)
    return existed


# ── Tool registration ──────────────────────────────────────────

def _sanitize(part: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(part))[:40]


def registered_name(server: str, tool: str) -> str:
    return f"{TOOL_PREFIX}_{_sanitize(server)}_{_sanitize(tool)}"


def register_server_tools(server: str, spec: dict, tools: list,
                          call_fn) -> list[str]:
    """Register a server's tools into the agent loop registry.

    `tools` are MCP tool listings (objects or dicts with name /
    description / inputSchema). Returns the registered names so the
    caller can unregister them on disconnect. A name that would clobber
    an existing tool from elsewhere is skipped, never overwritten.
    """
    auto_tools = set(spec.get("auto_tools", []))
    registered: list[str] = []
    for tool in tools:
        t_name = _field(tool, "name", "")
        if not t_name:
            continue
        full_name = registered_name(server, t_name)
        if full_name in agent_loop.TOOLS:
            continue
        schema = _field(tool, "inputSchema", None)
        if not isinstance(schema, dict) or schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        description = _field(tool, "description", "") or t_name
        agent_loop.register_tool(agent_loop.Tool(
            name=full_name,
            description=f"[external MCP server '{server}'] {description}"[:1024],
            parameters=schema,
            risk="auto" if t_name in auto_tools else "staged",
            handler=_make_handler(server, t_name, call_fn),
        ))
        registered.append(full_name)
    return registered


def _make_handler(server: str, tool: str, call_fn):
    def handler(**kwargs):
        egress_log.declare(
            f"mcp:{server}",
            f"external MCP tool '{tool}' — model-chosen arguments handed "
            "to a third-party tool process",
            port=0,
        )
        return call_fn(server, tool, kwargs)
    return handler


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def result_text(result: Any) -> str:
    """Flatten an MCP call result into text for the model."""
    parts = []
    for item in _field(result, "content", None) or []:
        kind = _field(item, "type", "")
        if kind == "text":
            parts.append(_field(item, "text", "") or "")
        else:
            parts.append(f"[{kind or 'non-text'} content omitted]")
    text = "\n".join(p for p in parts if p) or "(empty result)"
    if _field(result, "isError", False):
        return f"Tool error: {text}"
    return text


# ── Connection manager ─────────────────────────────────────────

class MCPManager:
    """Owns one event-loop thread and one task per connected server."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        # All three dicts are only mutated from the loop thread once a
        # connection exists; reads from request threads are dict lookups.
        self.sessions: dict[str, Any] = {}
        self.server_tools: dict[str, list[str]] = {}
        self.stop_events: dict[str, asyncio.Event] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.errors: dict[str, str] = {}

    # -- loop plumbing --

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._loop.run_forever,
                    daemon=True,
                    name="vaultmind-mcp",
                )
                self._thread.start()
            return self._loop

    def _run(self, coro, timeout: float):
        future = asyncio.run_coroutine_threadsafe(coro, self._ensure_loop())
        return future.result(timeout=timeout)

    # -- lifecycle --

    def connect(self, name: str, spec: dict) -> dict:
        """Connect one server (sync). Returns {"tools": [...]} or raises."""
        if not MCP_AVAILABLE:
            raise RuntimeError(
                "The 'mcp' package is not installed. "
                "Run: pip install 'mcp>=1.2,<2.0'"
            )
        if name in self.sessions:
            self.disconnect(name)
        self.errors.pop(name, None)
        try:
            tools = self._run(self._start_server(name, spec),
                              timeout=CONNECT_TIMEOUT + 5)
            return {"tools": tools}
        except Exception as e:
            self.errors[name] = str(e)
            raise

    async def _start_server(self, name: str, spec: dict) -> list[str]:
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        task = asyncio.ensure_future(self._server_task(name, spec, ready))
        self.tasks[name] = task
        try:
            return await asyncio.wait_for(asyncio.shield(ready), CONNECT_TIMEOUT)
        except asyncio.TimeoutError:
            task.cancel()
            raise RuntimeError(
                f"Server {name!r} did not initialize within {CONNECT_TIMEOUT:.0f}s."
            )

    async def _server_task(self, name: str, spec: dict,
                           ready: asyncio.Future) -> None:
        """One task owns the whole connection, entry to exit.

        The SDK's stdio context managers must be entered and exited in
        the same task, so teardown happens by setting the stop event and
        letting this task unwind — never by closing from outside.
        """
        registered: list[str] = []
        try:
            params = StdioServerParameters(
                command=spec["command"],
                args=list(spec.get("args", [])),
                env=(spec.get("env") or None),
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listing = await session.list_tools()
                    registered = register_server_tools(
                        name, spec, listing.tools, self.call
                    )
                    stop = asyncio.Event()
                    self.sessions[name] = session
                    self.server_tools[name] = registered
                    self.stop_events[name] = stop
                    if not ready.done():
                        ready.set_result(registered)
                    await stop.wait()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.errors[name] = str(e)
            if not ready.done():
                ready.set_exception(
                    RuntimeError(f"Could not connect to {name!r}: {e}")
                )
        finally:
            self.sessions.pop(name, None)
            self.stop_events.pop(name, None)
            self.tasks.pop(name, None)
            for tool_name in self.server_tools.pop(name, registered) or []:
                agent_loop.unregister_tool(tool_name)

    def disconnect(self, name: str) -> None:
        loop, task = self._loop, self.tasks.get(name)
        stop = self.stop_events.get(name)
        if loop is None or task is None:
            return
        if stop is not None:
            loop.call_soon_threadsafe(stop.set)
        else:
            loop.call_soon_threadsafe(task.cancel)
        try:
            asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(asyncio.shield(task), 10), loop
            ).result(timeout=15)
        except Exception:
            pass  # teardown is best-effort; the task's finally unregisters tools

    def connect_all(self) -> None:
        """Connect every enabled configured server; failures become status."""
        for name, spec in load_config()["servers"].items():
            if not spec.get("enabled", True):
                continue
            try:
                self.connect(name, spec)
                print(f"🔌 MCP server '{name}' connected "
                      f"({len(self.server_tools.get(name, []))} tools, staged by default)")
            except Exception as e:
                print(f"⚠️  MCP server '{name}' failed to connect: {e}")

    def shutdown(self) -> None:
        for name in list(self.tasks):
            self.disconnect(name)
        with self._lock:
            loop, self._loop = self._loop, None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)

    # -- calls --

    def call(self, server: str, tool: str, arguments: dict) -> str:
        session = self.sessions.get(server)
        if session is None:
            return (f"Tool error: MCP server '{server}' is not connected. "
                    "Reconnect it from Settings → MCP servers.")
        try:
            result = self._run(
                session.call_tool(tool, arguments or {}), timeout=CALL_TIMEOUT
            )
        except Exception as e:
            return f"Tool error: call to '{tool}' on '{server}' failed: {e}"
        return result_text(result)

    # -- status --

    def status(self) -> dict:
        servers = {}
        for name, spec in load_config()["servers"].items():
            servers[name] = {
                "command": spec["command"],
                "args": spec.get("args", []),
                "enabled": spec.get("enabled", True),
                "connected": name in self.sessions,
                "tools": self.server_tools.get(name, []),
                "auto_tools": spec.get("auto_tools", []),
                "error": self.errors.get(name),
            }
        return {"mcp_available": MCP_AVAILABLE, "servers": servers}


manager = MCPManager()

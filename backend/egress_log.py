"""Observed record of every outbound network connection this process makes.

The privacy panel used to return a hardcoded empty list of external
connections and the string "never for your personal data", regardless of
what had actually happened. That is the worst kind of bug to have in a
privacy product: not a leak, but a false assurance that no one can check.

The obvious fix is to log at each call site. That is only ever as honest as
the bookkeeping around it — a call site added later, or a dependency making
its own requests, silently goes unrecorded. `ddgs` and the Google API client
both open their own connections, and neither is under this codebase's
control.

So this instruments `socket.connect` instead. Any library that opens sockets
through CPython passes through it — `requests`, `httpx`, `urllib3`, the
Google API client, the Ollama client — which means the panel reports what
the process did, not what this file believes it did.

There is one important exception, and it is the reason this docstring is
long.

`ddgs`, which performs web search, uses `primp` — a Rust HTTP client that
opens its own sockets through Rust's networking stack, never touching
CPython's socket module. Two searches returning six results produce exactly
zero observed connections. A panel that only reported socket observations
would therefore show "0 external connections" while the user's query text
was being sent to a search engine. That is worse than the hardcoded lie it
replaced, because it looks like evidence.

So connections come from two sources and are labelled as such:

  observed  — seen at the socket layer. Comprehensive for anything using
              CPython sockets. This is the trustworthy kind.
  declared  — reported by the call site, because the library performing the
              request cannot be observed from here. Only as good as the
              instrumentation around it, and marked so nobody mistakes it
              for proof.

Anything that is `observed` can be checked against `tcpdump` or Little Snitch
and should agree. If it ever disagrees, this module is wrong. Anything
`declared` is a statement of intent, not a measurement, and the UI says so.

Deliberately connection-level, not byte-level: counting bytes means wrapping
send/recv on every socket, which is invasive and slow. Note also that HTTP
keep-alive means one connection can carry many requests, so connection
counts understate request volume — they are a record of who was talked to,
not how much.
"""

import ipaddress
import socket
import threading
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps

MAX_EVENTS = 500
MAX_DISCLOSURES = 200

_events: deque = deque(maxlen=MAX_EVENTS)

# Passages handed to a connected AI client (currently the MCP server).
# Tracked separately from connections because the mechanism is different:
# this process does not make the outbound request — it hands text to a
# client, which then sends it wherever that client sends things. No socket
# monitor can see that, so the only honest record is what we handed over.
_disclosures: deque = deque(maxlen=MAX_DISCLOSURES)

_lock = threading.Lock()
_local = threading.local()
_installed = False

# Populated by the getaddrinfo hook so connections can be labelled with the
# hostname that was resolved rather than a bare address.
_ip_to_host: dict[str, str] = {}

_SESSION_START = datetime.now(timezone.utc)


# ── Purpose labelling ──────────────────────────────────────────

@contextmanager
def purpose(name: str):
    """Tag connections opened in this thread while the block is active.

    Best-effort context, not a security boundary — a connection opened on a
    different thread inside the block will not pick up the label. It is
    still recorded, just as "unlabelled".
    """
    previous = getattr(_local, "purpose", None)
    _local.purpose = name
    try:
        yield
    finally:
        _local.purpose = previous


def labelled(name: str):
    """Decorator form of `purpose`, for whole functions that do network work."""
    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with purpose(name):
                return fn(*args, **kwargs)
        return wrapper
    return decorate


def _current_purpose() -> str:
    return getattr(_local, "purpose", None) or "unlabelled"


# ── Classification ─────────────────────────────────────────────

def _is_local(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host in ("localhost", "localhost.localdomain")
    return bool(ip.is_loopback or ip.is_private or ip.is_link_local
                or ip.is_unspecified)


def _record(host: str, port: int) -> None:
    hostname = _ip_to_host.get(host, host)
    event = {
        "at": datetime.now(timezone.utc).isoformat(),
        "host": hostname,
        "address": host,
        "port": port,
        "scope": "local" if _is_local(host) else "external",
        "purpose": _current_purpose(),
        "source": "observed",
    }
    with _lock:
        _events.append(event)


def disclose(tool: str, query: str, sources: list, characters: int,
             client: str = "MCP client") -> None:
    """Record vault text handed to a connected AI client.

    When Claude (or Cursor, or any MCP client) retrieves from the vault,
    the passages returned go wherever that client sends them — for a cloud
    model, that means off this machine. VaultMind does not make that
    request and cannot observe it, so this records the one thing that is
    knowable: exactly what was handed over, and in answer to what.

    This is the difference between MCP and uploading your files. The corpus
    stays here; specific passages leave, one question at a time, and this
    is the log of which.
    """
    with _lock:
        _disclosures.append({
            "at": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "query": (query or "")[:300],
            "sources": list(sources)[:20],
            "characters": int(characters),
            "client": client,
        })


def disclosure_summary() -> dict:
    with _lock:
        items = list(_disclosures)
    sources = sorted({s for d in items for s in d["sources"]})
    return {
        "retrievals": len(items),
        "characters_shared": sum(d["characters"] for d in items),
        "sources_touched": sources,
        "last_at": items[-1]["at"] if items else None,
        "note": (
            "Passages handed to a connected AI client. If that client is a "
            "cloud model, this text reached it. Your documents themselves "
            "were not uploaded — only these excerpts, in answer to these "
            "questions."
        ),
    }


def recent_disclosures(limit: int = 50) -> list[dict]:
    with _lock:
        return list(_disclosures)[-limit:]


def declare(host: str, purpose_text: str, port: int = 443) -> None:
    """Record a connection this module cannot observe.

    For libraries that bypass CPython's socket layer — currently `primp`,
    used by `ddgs` for web search. Call this at the point the request is
    made. It is marked `declared` rather than `observed` so the interface
    can be clear that it is self-reported, not measured.
    """
    with _lock:
        _events.append({
            "at": datetime.now(timezone.utc).isoformat(),
            "host": host,
            "address": None,
            "port": port,
            "scope": "external",
            "purpose": purpose_text,
            "source": "declared",
        })


# ── Installation ───────────────────────────────────────────────

def install() -> None:
    """Patch socket so every outbound connection is recorded.

    Must run before dependencies capture their own references to these
    functions, so main.py calls it at import time.
    """
    global _installed
    if _installed:
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def _note_address(address) -> None:
        # Unix sockets and odd address families are not interesting here.
        if isinstance(address, tuple) and len(address) >= 2:
            try:
                _record(str(address[0]), int(address[1]))
            except Exception:
                pass

    def connect(self, address):
        _note_address(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        _note_address(address)
        return real_connect_ex(self, address)

    def getaddrinfo(host, port, *args, **kwargs):
        results = real_getaddrinfo(host, port, *args, **kwargs)
        if host:
            for entry in results:
                sockaddr = entry[4]
                if isinstance(sockaddr, tuple) and sockaddr:
                    _ip_to_host.setdefault(str(sockaddr[0]), str(host))
        return results

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo
    _installed = True


# ── Reporting ──────────────────────────────────────────────────

def snapshot() -> dict:
    """Summarise what this process has connected to since it started."""
    with _lock:
        events = list(_events)

    external, local = {}, {}
    for e in events:
        bucket = external if e["scope"] == "external" else local
        key = e["host"]
        entry = bucket.setdefault(key, {
            "host": key,
            "connections": 0,
            "purposes": set(),
            "source": e["source"],
            "first_seen": e["at"],
            "last_seen": e["at"],
        })
        entry["connections"] += 1
        entry["purposes"].add(e["purpose"])
        entry["last_seen"] = e["at"]
        # A host seen both ways is only as trustworthy as its weakest source.
        if e["source"] == "declared":
            entry["source"] = "declared"

    def finish(bucket):
        out = []
        for entry in bucket.values():
            entry = dict(entry)
            entry["purposes"] = sorted(entry["purposes"])
            out.append(entry)
        return sorted(out, key=lambda x: -x["connections"])

    external_list = finish(external)
    return {
        "monitoring": _installed,
        "session_started": _SESSION_START.isoformat(),
        "external": external_list,
        "local": finish(local),
        "external_connection_count": sum(
            e["connections"] for e in external.values()
        ),
        "declared_hosts": [
            e["host"] for e in external_list if e["source"] == "declared"
        ],
        "disclosures": disclosure_summary(),
        "truncated": len(events) >= MAX_EVENTS,
        "method": (
            "Connections marked 'observed' are seen at the socket layer and "
            "should match tcpdump. Connections marked 'declared' are reported "
            "by the code making the request, because the library performing "
            "it (currently primp, used for web search) opens sockets outside "
            "Python and cannot be observed here. Counts are connections, not "
            "requests — keep-alive means one connection can carry many."
        ),
    }


def recent(limit: int = 100) -> list[dict]:
    with _lock:
        return list(_events)[-limit:]

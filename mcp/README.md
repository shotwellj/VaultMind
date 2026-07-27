# VaultMind MCP server

Give Claude access to your private files without giving Anthropic your
private files.

Connects [VaultMind](https://github.com/shotwellj/VaultMind) — a local-first
document vault running on your own machine — to Claude Desktop, Claude Code,
Cursor, or any other MCP client.

---

## What actually leaves your machine

Worth being exact, because "private" gets used loosely.

**Your documents are never uploaded.** They stay on disk, indexed locally,
embedded by a local model. This server never sends a file anywhere.

**Passages do leave.** When Claude calls `vault_search`, the excerpts it gets
back are sent by Claude to Anthropic's model, because that is how Claude
works. The same is true of any cloud MCP client.

So this is not "nothing leaves." It is: *you choose what leaves, one question
at a time, and you can see exactly what it was.* Every passage handed over is
recorded — open the VaultMind privacy panel, or:

```bash
curl -s -H "X-VaultMind-Token: $(cat backend/.vaultmind_token)" \
  http://127.0.0.1:8000/mcp/disclosures
```

That log is the difference between this and uploading a folder to a chatbot.

---

## Install

Requires VaultMind running locally (`bash start.sh`).

Add to your MCP client config — for Claude Desktop, that is
`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vaultmind": {
      "command": "uvx",
      "args": ["vaultmind-mcp"]
    }
  }
}
```

Restart the client. Ask it something about your own documents.

Using `pipx` or a plain virtualenv instead:

```bash
pipx install vaultmind-mcp
```

```json
{
  "mcpServers": {
    "vaultmind": { "command": "vaultmind-mcp" }
  }
}
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `VAULTMIND_URL` | `http://127.0.0.1:8000` | Where the backend is listening. |
| `VAULTMIND_TOKEN` | *(read from disk)* | Access token. Normally found automatically. |
| `VAULTMIND_DATA_DIR` | *(auto)* | Where to look for `.vaultmind_token`. |

The token is written by the backend on first run. This server checks
`VAULTMIND_DATA_DIR`, then the Electron app's data directory, then a source
checkout — so in most setups nothing needs configuring.

---

## Tools

**`vault_search(query, limit=5)`** — the one to reach for. Semantic search
across everything indexed, returning excerpts with their source. Capped at
1,200 characters per passage and 10 passages, because the amount returned is
the amount that reaches the model.

**`vault_list_sources()`** — names and chunk counts of indexed documents. No
content. Good for "what do you have access to".

**`vault_get_document(source, max_chars=6000)`** — more of a single document,
when an excerpt is not enough. Capped, so a client cannot drain the vault one
file at a time. Prefer `vault_search`.

---

## Design

This process is a thin shim. It holds no state, and does not touch ChromaDB
or Ollama directly — it calls the running VaultMind backend over localhost.

That is deliberate. Retrieval logic stays in one place rather than drifting
between two implementations, and, more importantly, nothing can be retrieved
without being logged. A version that read the vector store directly would
have been slightly faster and would have made the disclosure log a lie.

---

## Troubleshooting

**"VaultMind is not running"** — start it with `bash start.sh`. If it listens
somewhere other than port 8000, set `VAULTMIND_URL`.

**"Could not find the VaultMind access token"** — the error lists the paths
checked. Set `VAULTMIND_TOKEN` or `VAULTMIND_DATA_DIR` to point at the right
one.

**"VaultMind rejected the access token"** — the token is regenerated if the
data directory is cleared. Restart your MCP client so it re-reads the file.

**Empty results for something you know is indexed** — check the document is
listed by `vault_list_sources`, then try more specific wording. If the vault
is large and searches consistently miss, run
`scripts/calibrate_threshold.py` in the main repo to re-derive the relevance
cutoff for your corpus.

---

Apache 2.0. Part of [VaultMind](https://github.com/shotwellj/VaultMind).

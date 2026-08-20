# VaultMind

**Chat with your documents, email, and the web — entirely on your own machine.**

No API keys. No cloud. No subscription. Everything runs locally via [Ollama](https://ollama.ai).

![VaultMind screenshot](docs/screenshot.png)

---

## Why VaultMind

Every other "chat with your docs" tool sends your documents to OpenAI, Anthropic, or some other cloud. VaultMind doesn't. The LLM runs on your hardware, the vector database lives on your disk, and **your documents are never uploaded anywhere.**

### What does leave your machine

Being precise about this, because "100% local" is easy to say and easy to get wrong:

| | Leaves your machine? |
|---|---|
| Your documents, their text, their embeddings | Never |
| The model, and every answer it generates | Never — inference is local |
| Your question, in **vault mode** | No |
| Your question, in **agent mode** | Yes — sent to DuckDuckGo as a search query |
| Pages VaultMind fetches for you | Yes — it requests the URL, like a browser |
| Gmail / Notion sync | Yes — it authenticates to those APIs to pull your data down |

So: the private corpus stays put, and the network is used only for search and for connectors you explicitly turn on. Agent mode is the one to know about — when it is on, your question text reaches a search engine. The privacy firewall (Settings → Privacy) strips some identifiers before searching, but it is regex-based unless you install spaCy, so treat it as a reduction in exposure rather than a guarantee. See `backend/requirements.txt`.

| | VaultMind | ChatGPT / Claude | PrivateGPT | Obsidian Copilot |
|---|---|---|---|---|
| 100% local | ✅ | ❌ | ✅ | ❌ |
| Gmail integration | ✅ | ❌ | ❌ | ❌ |
| Live web search | ✅ | ✅ | ❌ | ❌ |
| One-command setup | ✅ | ✅ | ❌ | ❌ |
| No API key needed | ✅ | ❌ | ✅ | ❌ |
| Electron Mac app | ✅ | — | ❌ | ❌ |

---

## Quick Start

**Prerequisite:** [Ollama](https://ollama.ai/download) installed and running.

```bash
git clone https://github.com/shotwellj/VaultMind.git
cd VaultMind
bash start.sh
```

That's it. `start.sh` pulls the models (~4.5 GB, one-time), installs Python deps, starts the backend, and opens `http://localhost:8000`.

### Docker (no Ollama install required)

```bash
docker compose up
```

Then open `http://localhost:8000`. Docker handles everything including Ollama.

---

## What You Can Do

**Drop in any file** — PDF, DOCX, TXT, Markdown, CSV — and ask questions across all of it.

**Paste any URL** — VaultMind fetches and indexes it instantly. Paste a job posting, a competitor's pricing page, a research paper.

**Connect Gmail** — OAuth into your inbox and VaultMind indexes your emails locally. Ask *"what did my lawyer say about the contract?"* or *"summarize my inbox"*.

**Connect Notion** — Paste your integration token and your workspace syncs automatically on a configurable schedule.

**Agent mode** — Toggle 🌐 Agent and VaultMind combines your private vault with live web search for questions your docs can't answer alone.

**Use it from Claude, Cursor, or Claude Code** — VaultMind ships an MCP server, so your assistant can search your vault without your documents being uploaded anywhere:

```json
{
  "mcpServers": {
    "vaultmind": { "command": "uvx", "args": ["vaultmind-mcp"] }
  }
}
```

Passages the assistant retrieves *do* reach whatever model it runs on — the difference from uploading your files is that you choose what leaves, one question at a time, and every passage is logged. See [mcp/README.md](mcp/README.md).

---

## How It Works

```
Files / URLs / Gmail / Notion
           │
           ▼
    Text extraction
  (pypdf · python-docx · BS4)
           │
           ▼
   150-word chunks, 20-word overlap
           │
           ▼
  nomic-embed-text (local, via Ollama)
           │
           ▼
    ChromaDB on disk
           │
           ▼
  Query → embed → top-k similarity search
           │
           ▼
  Mistral / Llama / Phi / Gemma (your choice)
           │
           ▼
      Streamed answer
```

Every step above runs on your machine. The API is FastAPI on `localhost:8000`. The UI is a single HTML file — no framework, no build step, no CDN.

---

## Stack

| Layer | Tool |
|---|---|
| LLM + embeddings | [Ollama](https://ollama.ai) |
| Vector store | [ChromaDB](https://www.trychroma.com) |
| Backend | FastAPI + streaming SSE |
| Frontend | Vanilla JS — zero dependencies |
| Document parsing | pypdf, python-docx, BeautifulSoup |
| Gmail | Google OAuth 2.0 (readonly) |
| Desktop app | Electron (Mac) |

---

## Supported Models

Switch models any time from the sidebar dropdown. All run locally via Ollama.

- **Mistral 7B** — fast, good all-rounder (default)
- **Llama 3.2** — strong reasoning
- **Phi-3 Mini** — lightweight, great on older hardware
- **Gemma 2** — Google's open model
- **Qwen 2.5** — strong on technical content
- **DeepSeek R1** — best for complex analysis

---

## Mac App

VaultMind ships as a native Electron app — no Terminal required.

```bash
npm install
npm start          # dev mode
bash build-app.sh  # builds distributable .dmg
```

First launch automatically creates a Python virtualenv and installs dependencies. Subsequent launches skip straight to the app. User data lives in `~/Library/Application Support/VaultMind/data` — safe across app updates.

---

## Access From Your Phone

VaultMind runs in any browser. On your local network:

```bash
ipconfig getifaddr en0   # find your Mac's IP
# open http://192.168.x.x:8000 on your phone
```

From anywhere via [Tailscale](https://tailscale.com) (free, 5 min setup):

1. Install Tailscale on your Mac and phone, sign in with the same account
2. Open `http://100.x.x.x:8000` from anywhere — stays completely private

Tap **Add to Home Screen** in Safari to install as a PWA.

**Reaching it from another device takes two steps.** By default the backend binds to `127.0.0.1`, so nothing outside your Mac can connect. To change that:

```bash
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
```

Every endpoint requires a token, generated on first run and stored in `backend/.vaultmind_token`. Opening the app in a browser hands the page that token automatically. Prefer Tailscale over plain LAN — on a shared or public network, `0.0.0.0` exposes the port to everyone on it, and the token is then the only thing between them and your documents.

---

## What's Built

- [x] PDF, DOCX, TXT, MD, CSV upload
- [x] URL ingestion (scrape any page)
- [x] Gmail OAuth — index inbox locally
- [x] Notion sync — auto-polls on configurable schedule
- [x] Agent mode — vault + live web search
- [x] Inbox digest — AI-ranked email summary
- [x] 6 local model choices
- [x] Agent runs — multi-step tool loop with approval gates (`POST /agent/run`)
- [x] Live run streaming — watch each step in the 🤖 Runs tab, approve or reject inline
- [x] External MCP servers — plug any MCP tool into the agent loop, staged by default (`/agent/mcp`)
- [x] Electron Mac app
- [x] Docker support
- [x] Mobile-responsive PWA

## What's Next

- [ ] Slack integration
- [ ] WhatsApp conversation export
- [ ] Bulk URL ingestion
- [ ] Timeline view — *"what happened in March?"*
- [ ] MCP server — use VaultMind as context inside Cursor / VS Code

---

## Contributing

Apache 2.0. PRs welcome.

```bash
# Backend dev mode (hot reload)
cd backend && uvicorn main:app --reload --port 8000

# Frontend is at http://localhost:8000 — edit frontend/index.html directly

# Tests — these run in CI on every push
pip install pytest && python -m pytest tests -v
```

`tests/test_regressions.py` covers bugs that shipped in v1.0.0: the retrieval
threshold, the shadowed `/agent` route, agent-tool path confinement, and the
SSRF guard. If you touch any of those, that file should tell you.

Open an issue for bugs. Open a discussion for feature ideas.

---

*Built by [Jason Shotwell](https://github.com/airblackbox). Part of the [AIR Blackbox](https://airblackbox.ai) ecosystem.*

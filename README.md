# VaultMind

**Chat with your documents, email, and the web — entirely on your own machine.**

No API keys. No cloud. No subscription. Everything runs locally via [Ollama](https://ollama.ai).

![VaultMind screenshot](docs/screenshot.png)

---

## Why VaultMind

Every other "chat with your docs" tool sends your data to OpenAI, Anthropic, or some other cloud. VaultMind doesn't. The LLM runs on your hardware. The vector database lives on your disk. Nothing is transmitted anywhere.

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
git clone https://github.com/airblackbox/VaultMind.git
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

100% local. The API is FastAPI on `localhost:8000`. The UI is a single HTML file — no framework, no build step.

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

---

## What's Built

- [x] PDF, DOCX, TXT, MD, CSV upload
- [x] URL ingestion (scrape any page)
- [x] Gmail OAuth — index inbox locally
- [x] Notion sync — auto-polls on configurable schedule
- [x] Agent mode — vault + live web search
- [x] Inbox digest — AI-ranked email summary
- [x] 6 local model choices
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
```

Open an issue for bugs. Open a discussion for feature ideas.

---

*Built by [Jason Shotwell](https://github.com/airblackbox). Part of the [AIR Blackbox](https://airblackbox.ai) ecosystem.*

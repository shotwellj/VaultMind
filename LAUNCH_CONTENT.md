# VaultMind Launch Content

Post these Thursday morning, 8-10am EST for best HN traction.

---

## Hacker News — Show HN

**Title:**
```
Show HN: VaultMind – chat with your local documents using Ollama, no cloud
```

**Comment to post immediately after (your first comment):**
```
Built this over a couple of days as part of the AIR Blackbox ecosystem
(https://airblackbox.ai). Same core thesis: AI that stays on your machine.

Stack is straightforward — FastAPI backend, ChromaDB for vector search,
nomic-embed-text for embeddings, llama3.2 for inference. All running
locally via Ollama. The UI is a single HTML file.

Drop in a PDF, ask questions, get streamed answers. Conversation memory
works across follow-ups. File management built in.

It's early and rough around the edges. Happy to answer questions about
the RAG pipeline or the architecture decisions.

What I'm building next: .txt/.md/.docx support, Gmail Takeout import,
and eventually a one-command Docker setup so anyone can run it.

Repo: https://github.com/air-blackbox/vaultmind
```

---

## Twitter/X Thread

```
1/
Your documents contain things you'd never upload to ChatGPT.

Medical records. Contracts. Tax returns. Personal notes.

I built VaultMind — a local RAG system that lets you chat with your
documents using Ollama. Nothing leaves your machine.

Here's how it works:

2/
Stack:
- FastAPI backend
- ChromaDB for vector search
- nomic-embed-text for local embeddings
- llama3.2 for inference
- Single HTML file for the UI

Clone it, pull the models, run uvicorn. That's it.

3/
The RAG pipeline:

PDF → pypdf → 150-word chunks with overlap
→ nomic-embed-text embeds each chunk
→ stored in ChromaDB on disk

Query → embed the question → similarity search
→ top 6 chunks as context → llama3.2 streams the answer

4/
Conversation memory works across follow-ups.
Answers stream token by token.
Files can be added or removed from the index anytime.
ChromaDB persists between restarts.

It's rough but it works.

5/
Part of the AIR Blackbox ecosystem.

Same philosophy as the compliance scanner:
your data stays on your hardware.

Apache 2.0. PRs welcome.

→ https://github.com/air-blackbox/vaultmind
```

---

## LinkedIn

```
Shipped VaultMind this week — a local document AI that runs entirely on your machine.

The idea is simple: you have documents you'd never upload to ChatGPT. Medical records, contracts, financial statements, personal notes. But you still want to be able to ask questions about them in plain English.

VaultMind indexes your PDFs locally using ChromaDB for vector search and Ollama for embeddings and inference. The whole thing runs on localhost. Nothing hits a server.

Ask "what were my blood pressure readings last March?" or "what are the payment terms in this contract?" and get a streamed answer backed by the actual source documents.

It's early — PDFs only, no mobile app, no hardware bundle yet. But the core pipeline works and it's open source.

Part of the AIR Blackbox project (airblackbox.ai). Same local-first philosophy.

Repo in the comments.

#OpenSource #LocalAI #Privacy #RAG #Ollama
```

---

## Reddit Posts

**r/LocalLLaMA title:**
```
Built a local RAG system for personal documents — Ollama + ChromaDB + FastAPI, zero cloud
```

**r/selfhosted title:**
```
VaultMind – chat with your documents using local LLMs (no cloud, no accounts, single HTML UI)
```

Both posts: link to GitHub, paste the Quick Start section, mention it's Apache 2.0.

# VaultMind

**A private knowledge operating system. Your data, your machine, your rules.**

VaultMind indexes everything you know — documents, spreadsheets, web pages, exported data from any app — and lets you query all of it in plain English using a local LLM. Nothing leaves your machine. No subscription. No vendor.

Part of the [AIR Blackbox](https://airblackbox.ai) ecosystem — privacy-first AI tooling.

---

## The Idea

Every SaaS product you pay for is three things: a database, a UI, and a way to query your data. You're paying Salesforce $150/month because it can answer *"who are my hot leads"* — but YOUR data is doing all the work.

VaultMind cuts them out. Export your data from any app, drop it in, and ask questions in plain English. Your data stays on your hardware. Forever.

**Stop renting access to your own data.**

---

## What You Can Do With It

### Replace SaaS query layers
- **Salesforce / HubSpot** → export contacts + notes as CSV, ask *"who haven't I followed up with in 30 days?"*
- **QuickBooks / Intuit** → export transactions, ask *"what's my average monthly revenue this year?"*
- **Notion** → export your workspace, ask *"what did I decide about X last month?"*
- **Gmail** → Google Takeout export, ask *"what did my lawyer say about the contract?"*

### Build a private intelligence database
- Drop in competitor research, market reports, industry docs
- Index your own playbooks, SOPs, methodologies
- Query your own expertise: *"what's my sourcing strategy for passive candidates?"*
- Feed it URLs — scrape any webpage and make it searchable instantly

### Use URL ingestion as a research tool
Paste any URL and VaultMind indexes it. Use cases:
- **Recruiting**: scrape LinkedIn profiles, job boards, company pages — ask *"find me data engineers in Irvine"* using your own sourcing playbooks as context
- **Sales**: index a prospect's website before a call — ask *"what are their main products and who do they sell to?"*
- **Research**: index articles, docs, competitor pages — ask anything across all of them
- **Due diligence**: build a private research file on any company

---

## Quick Start

**Requirements**: [Ollama](https://ollama.ai) installed and running

```bash
# 1. Clone
git clone https://github.com/air-blackbox/vaultmind.git
cd vaultmind

# 2. Run the launcher — handles everything automatically
bash start.sh
```

`start.sh` checks for Ollama, pulls the required models, installs dependencies, starts the backend, and opens the UI. One command, done.

---

## Supported Input Types

| Type | Formats | How to get your data |
|------|---------|---------------------|
| Documents | PDF, DOCX, TXT, MD | Direct upload |
| Spreadsheets | CSV | Export from any app |
| Web pages | Any URL | Paste and index instantly |
| CRM data | CSV export | Salesforce, HubSpot, etc. |
| Email | Export to TXT/MD | Gmail Takeout, Mail export |
| Financial | CSV export | QuickBooks, Stripe, etc. |

---

## Architecture

```
Any file or URL
      │
      ▼
Text extraction (pypdf / python-docx / BeautifulSoup)
      │
      ▼
chunk_text() — 150-word chunks, 20-word overlap
      │
      ▼
nomic-embed-text (Ollama) — local embeddings, no API
      │
      ▼
ChromaDB — persistent vector store on disk
      │
      ▼
User query → embed → similarity search → top 6 chunks
      │
      ▼
llama3.2 (Ollama) — streamed response with conversation memory
```

100% local. The API is FastAPI on `localhost:8000`. The UI is a single HTML file.

---

## Stack

| Layer | Tool |
|-------|------|
| LLM inference | [Ollama](https://ollama.ai) (llama3.2) |
| Embeddings | nomic-embed-text via Ollama |
| Vector store | [ChromaDB](https://www.trychroma.com) — persists on disk |
| Backend | FastAPI + streaming responses |
| Frontend | Vanilla JS — no framework, no build step |
| Document parsing | pypdf, python-docx, BeautifulSoup |

---

## Why Local-First

Cloud AI tools require you to upload your data to their servers. Your contracts, your financials, your candidate research, your private notes — on someone else's machine, under someone else's terms of service.

VaultMind doesn't work that way. The models run on your hardware via Ollama. The vector database lives on your disk. Nothing is transmitted anywhere.

Same philosophy as [AIR Blackbox](https://airblackbox.ai): your data stays where you put it.

---

## Roadmap

- [ ] `.mbox` / Gmail Takeout email ingestion
- [ ] Apple Health export parsing
- [ ] Bulk URL ingestion (paste a list, index them all)
- [ ] Timeline view — *"what happened in March 2025?"*
- [ ] One-command Docker setup
- [ ] Multi-user / household support
- [ ] Voice input

---

## Contributing

Apache 2.0. PRs welcome. Open an issue if something breaks.

```bash
cd backend && uvicorn main:app --reload --port 8000
```

---

*Part of the [AIR Blackbox](https://github.com/air-blackbox) ecosystem.*

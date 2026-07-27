# Roadmap

Where VaultMind is going, and what it is deliberately not going to be.

Dates are intent, not commitment — this is maintained by one person.

---

## What VaultMind is

A personal AI that runs on your own machine, for the things you would never
paste into a cloud chatbot. Your medical records, your comp history, your
legal documents, your inbox.

## What it is becoming

Everything in this category asks you to take privacy on faith — "it runs
locally," believe it or don't. VaultMind is being built so you don't have to
take our word for it. The direction is **provable** privacy: a receipt for
every session showing exactly what left your machine and what didn't, that
you can verify yourself.

---

## v1.1: Truth — complete

Subtraction and honesty. No new features.

- [x] **Honest privacy dashboard.** It reported a hardcoded empty list of
      external connections regardless of what actually happened. It now
      reports real per-session network activity, observed at the socket
      layer, that you can check against `tcpdump` — plus `/privacy/connections`
      for the raw log. Connections the code cannot observe (web search runs
      through a Rust HTTP client that bypasses Python sockets) are labelled
      "reported" rather than "observed", so a self-report is never shown as
      a measurement.
- [x] **Delete what was never wired up.** 5,666 lines of modules that no
      code path reached — an unenforced RBAC layer, a CUDA fine-tuning
      pipeline shipped inside a Mac app, an unmounted sync protocol, and a
      mobile app calling endpoints that do not exist. See *Removed* below.
- [x] **Post-mortem on the retrieval bug.** Vault retrieval silently matched
      nothing for months because the relevance threshold was written for one
      distance metric and the collection used another. Fixed, with tests.
      Written up in [docs/posts/silent-rag-failure.md](posts/silent-rag-failure.md).

## v1.2: MCP server — complete

Use your private vault as context inside Claude Desktop, Claude Code, or
Cursor, without uploading your documents anywhere.

- [x] `vault_search`, `vault_list_sources`, `vault_get_document` over stdio
- [x] Copy-paste config block; installs with `uvx vaultmind-mcp`
- [x] Every passage handed to a client is logged, visible in the privacy
      panel and at `/mcp/disclosures`

The server is a thin shim that calls the local backend rather than reading
ChromaDB directly. Slightly slower, but retrieval logic stays in one place
and nothing can be retrieved without being recorded — a version that read
the vector store directly would have made the disclosure log a lie.

**To be precise about the claim:** your documents stay on your machine, but
passages returned to a cloud model *do* reach that model. The difference from
uploading your files is that you choose what leaves, one question at a time,
and you can see what it was.

## Now — v1.3: First run

Local AI apps die at install.

- [ ] Chat working in ~90 seconds with a small model, while the better one
      downloads in the background
- [ ] Onboarding that indexes something real and shows you what it found,
      instead of a blank prompt
- [ ] Three model choices (Fast / Balanced / Deep) instead of six coequal ones
- [ ] Fix or drop the Windows build — it currently ships an installer whose
      backend cannot start

## Later — v1.4: Receipts

- [ ] Hash-chained session record: every network call made and not made,
      every document read, every agent action taken
- [ ] Exportable, independently verifiable evidence bundle
- [ ] A bounty for anyone who can demonstrate a document leaving the machine

## Later — v1.5: What it knows about you

- [ ] A profile built from your own documents — employers, dates, recurring
      people, renewals and deadlines
- [ ] Proactive alerts: "your lease renews in 45 days"
- [ ] Click any sentence in an answer to see the exact source passage

---

## Removed

Being explicit, because deleting shipped code deserves an explanation.

| Removed | Why |
|---|---|
| `rbac.py` and `/auth/*` | Roles and permissions in a single-user local app. Never enforced on any endpoint; the token check it defined was applied nowhere. |
| `finetune_pipeline.py` | Generated a CUDA/Unsloth training script inside a macOS application. |
| `mobile_sync.py` | 603 lines, imported by nothing. |
| The mobile app | Called `/sync/pull`, `/sync/push` and `/sync/register`, none of which exist. Had never been run. |
| `mobile_alerts.py` | Push alerts for the app above. No push transport was ever implemented. |
| `photo_pipeline.py` | Backed only `/photos/*`, which raised on every call — it was passed argument names the function did not accept. Photo upload goes through `/upload-photo` and is unaffected. |
| `call_intel.py`, `contact_intel.py` | Unreachable from the interface; audio transcription and several core paths were explicit placeholders. |
| `LAUNCH_CONTENT.md`, `PUSH_TO_GITHUB.md` | Personal scratch notes that should never have been committed. |

The product had 112 registered routes and the interface used 24 of them.
Fewer, working features beat more, aspirational ones.

Still under review for a later pass: `export_layer.py`, `vertical_kit.py`,
`doc_compare.py` and the `/feedback/*` surface are also unreachable from the
UI, but each maps to something on the roadmap above, so they stay for now.

---

## Non-goals

Not planned, and saying so to save you the issue:

- **Multi-user, teams, shared workspaces.** It is a single-user local app.
- **Cloud sync.** Of anything, ever.
- **Model training or fine-tuning.**
- **Feature parity** with AnythingLLM, Open WebUI, or Msty. Different bet.
- **Telemetry**, unless it is opt-in, local-first, and readable before it is
  sent — which mostly means no.

---

## Influencing this

Open an issue describing what you were trying to do, not which feature you
want. Bug reports that include what you indexed and what you asked are worth
more than feature requests.

The clearest way to change the priorities above is to show that something is
broken for real use.

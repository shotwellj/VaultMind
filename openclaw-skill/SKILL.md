---
name: vaultmind
description: Query your private local document vault using VaultMind. Ask questions about indexed PDFs, Word docs, spreadsheets, URLs, and notes stored on your machine. Supports vault-only answers or combined vault + web search. Requires VaultMind running locally on port 8000.
version: 1.0.0
metadata:
  openclaw:
    requires:
      bins:
        - curl
---

# VaultMind — Private Document Intelligence

Query your personal knowledge vault using local AI. VaultMind indexes your private files and URLs on your machine using Ollama and ChromaDB. Nothing is sent to the cloud.

## When to use this skill

Use this skill when the user asks questions about:
- Their own documents (contracts, notes, reports, SOPs, resumes, health records, financial docs)
- Information they have previously saved or indexed
- Anything prefixed with "in my vault", "from my documents", "what does my contract say", "based on my files"
- Any question that sounds personal or private where a local knowledge base would be relevant

For general knowledge questions with no personal data angle, skip this skill and answer directly.

## How to call VaultMind

### Vault mode (answers from indexed documents only)

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"USER_QUESTION\", \"mode\": \"vault\"}"
```

### Agent mode (vault + live web search combined)

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"USER_QUESTION\", \"mode\": \"agent\"}"
```

## Response format

VaultMind returns JSON:

```json
{
  "answer": "The answer based on your indexed documents...",
  "sources": ["filename.pdf", "https://example.com/page"],
  "mode": "vault"
}
```

If VaultMind is not running or has no relevant docs indexed, it returns:
```json
{
  "answer": "I don't have any relevant information indexed for that question.",
  "sources": []
}
```

## Step-by-step instructions

1. Receive the user's question.
2. Determine whether to use vault mode or agent mode:
   - vault mode: question is clearly about personal documents ("what are the terms in my lease?")
   - agent mode: question needs both personal context AND live web info ("what's the current market rate for my role based on my resume?")
3. Replace USER_QUESTION with the user's actual question in the curl command above.
4. Run the curl command.
5. Parse the JSON response.
6. Return the `answer` field to the user.
7. If `sources` is non-empty, append: "Sources: [source1], [source2]"
8. If the answer indicates nothing is indexed, tell the user: "I didn't find anything relevant in your vault. Open VaultMind at frontend/index.html and index the relevant documents first."

## Check if VaultMind is running

```bash
curl -s http://localhost:8000/health
```

Returns `{"ready": true}` if everything is running. If not running, tell the user to run `bash start.sh` in their VaultMind folder.

## Example interactions

**User:** "What are the payment terms in my freelance contract?"
**Action:** vault mode query → return answer from indexed contract doc

**User:** "What's the current data engineer salary in LA based on my experience?"
**Action:** agent mode query → VaultMind searches resume (vault) + live salary data (web)

**User:** "Summarize my Nostalgic Skin Co. website audit"
**Action:** vault mode query → return answer from indexed audit doc

**User:** "What was my blood pressure last March?"
**Action:** vault mode query → return answer from indexed health records

## Setup

VaultMind must be running locally. If it isn't:
1. Tell the user to open Terminal
2. Navigate to their VaultMind folder
3. Run: `bash start.sh`
4. Wait for the setup wizard to complete
5. Retry the query

GitHub: https://github.com/airblackbox/VaultMind

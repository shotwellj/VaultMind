# My RAG app retrieved nothing for months and told me it was working

I shipped a local "chat with your documents" app. Documents indexed fine.
The chunk counter went up. Answers streamed back. It had a version number
and a release build.

Vault retrieval had never once worked.

Not "worked badly." Never returned a single chunk, for any question, from
any document, unless you explicitly pinned a file first. The bug was one
number, and every symptom it produced pointed somewhere else.

---

## The symptom I chased for months

Users — mostly me — would ask something about an indexed document and get
back a general web answer instead. My commit log is a monument to
misdiagnosis:

```
fix: stop web search from hijacking local vault queries
fix: agent mode web search + vault hallucination detection
Fix vault isolation: pin-to-file, routing cleanup, threshold tuning
```

Three separate attempts to stop web search from "taking over." I tuned
routing heuristics. I added a hallucination check. I built a pin-to-file
feature so you could force it to use a specific document.

The pin worked. That was the thing that kept me from finding the bug for
another two months, because it proved retrieval *could* work, so the problem
had to be in routing.

## The actual bug

Here is the retrieval code, condensed:

```python
def get_collection():
    return chroma.get_or_create_collection("vaultmind_vault")

RELEVANCE_THRESHOLD = 0.65   # ChromaDB L2 distance; tuned for personal docs

results = col.query(query_embeddings=[embedding], n_results=8)
relevant = [d for d, dist in zip(docs, dists) if dist < RELEVANCE_THRESHOLD]
```

That reads fine. There is even a comment explaining the units.

The comment is wrong, and it is wrong in the direction that makes the code
look considered. Chroma's default distance function is **squared L2**, and
`nomic-embed-text` produces vectors whose squared-L2 distances land in the
hundreds. Here are real measurements against a document I had just indexed:

| Query | Distance |
|---|---|
| Verbatim copy of the document's own text | **101.5** |
| A directly relevant question | **199.3** |
| Nonsense, unrelated to anything indexed | **568.2** |

The threshold was `0.65`.

The closest possible match — a document compared against *itself* — scored
156 times higher than the cutoff. Nothing could ever pass. `dist < 0.65` was
functionally `if False`.

There was even a fallback tier for when the strict pass found nothing:

```python
if not relevant:
    relevant = [d for d, dist in ... if dist < 0.85]
```

Also `if False`. I had written a fallback for a condition that was
permanent.

## Why every symptom pointed elsewhere

This is the part I think generalises.

**The failure was silent by construction.** Retrieval returning zero chunks
is indistinguishable from retrieval correctly determining that your vault
has nothing relevant. There is no error, no exception, no log line. The
system has no way to tell "I found nothing" from "there is nothing." Both
produce an empty list.

**The fallback made it look intentional.** Downstream of the empty result,
the code would fall through to web search — which is reasonable behaviour
when the vault genuinely has no answer. So the app did something plausible
every single time. It never crashed. It just quietly answered a different
question than the one you asked.

**The workaround masked the root cause.** Pinning a file skipped the
threshold check entirely and passed chunks straight through. It worked
perfectly. Every time I tested pinning, I confirmed to myself that
retrieval, embeddings, and chunking were all fine — which they were. So I
kept looking at the routing layer, which was innocent.

**The comment lied with confidence.** `# ChromaDB L2 distance; tuned for
personal docs` is exactly what you skim past. It names the right metric. It
implies someone measured something. I wrote it, and I believed it every time
I re-read that function.

I had almost certainly written `0.65` while thinking in cosine distance,
where the range is 0–2 and 0.65 is a sensible cutoff. Then I labelled it L2
and never checked.

## The fix

Two lines, plus a migration:

```python
VAULT_SPACE = {"hnsw:space": "cosine"}   # was: Chroma's default, squared L2

chroma.get_or_create_collection(VAULT_COLLECTION, metadata=VAULT_SPACE)
```

Chroma fixes the distance function when a collection is created, so existing
vaults have to be copied rather than reconfigured. The stored embeddings are
reused as-is, which means the migration costs no model calls and no
re-reading of anyone's files — a 653-chunk vault moved in under a second.

Then I set the threshold by measuring instead of guessing. Against a real
vault of PDFs and email:

| | Cosine distance |
|---|---|
| Questions the vault could answer | 0.28 – 0.41 |
| Questions it could not | 0.49 – 0.57 |

Clean separation, so the cutoff goes at **0.45**, in the gap. The old
fallback tier at `0.85` would have admitted every single unanswerable
question, which is its own lesson about tuning a number you have never
observed.

Those numbers are specific to one embedding model and one corpus, so I
shipped the measurement as a script rather than a constant. Point it at your
own vault, give it questions you know the answers to and questions you know
it cannot answer, and it reports the gap and a suggested threshold. When it
ran against my vault it independently suggested 0.44.

## What I'd tell anyone building retrieval

**Never write a distance threshold you have not printed.** One line — log
the distances for a query you know should match — would have caught this in
the first week. I never ran it, because the number in the code looked
reasonable and the app appeared to work.

**Assert on your own assumptions about units.** The distance metric of a
vector store is a load-bearing detail that lives in a default you did not
set. A test that indexes one document, queries it with its own text, and
asserts a hit would have failed from day one. That test exists now.

**Treat "no results" as a state worth reporting.** Empty retrieval and empty
vault are different conditions with identical signatures. The fix is not
clever ranking, it is telling the user which one happened.

**Be suspicious when a workaround works perfectly.** Pinning a file bypassed
the broken check and behaved flawlessly, and I read that as evidence the
pipeline was healthy. It was evidence that the pipeline was healthy *and
something was gating it*. A workaround that works is a bisection you have
already run — read it that way.

**Comments describing units are load-bearing and unverified.** Nothing
checks them. The wronger they are, the more they reassure.

---

The bug is fixed, the migration is automatic, and there are regression tests
now — 32 of the 33 fail against the previous commit, which is how I know
they test anything.

The code is at
[github.com/shotwellj/VaultMind](https://github.com/shotwellj/VaultMind).
The threshold calibration script is in `scripts/`, and if you have a RAG app
with a hardcoded distance cutoff, I would genuinely go and print your
distances before reading anything else today.

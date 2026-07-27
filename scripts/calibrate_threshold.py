#!/usr/bin/env python3
"""Re-derive the vault relevance threshold for your own vault.

The thresholds in main.py are cosine distances, and the right value depends on
the embedding model and on what you have indexed. Change the model and the old
number is meaningless.

Give this script questions your vault *should* answer and questions it should
not, and it reports the separation and a suggested threshold.

    python3 scripts/calibrate_threshold.py \
        --relevant "what does my lease say about pets" \
        --relevant "when does my contract renew" \
        --irrelevant "capital of Mongolia" \
        --irrelevant "how do I change a tyre"

With no arguments it uses a generic set, which is only a rough smoke test —
the answer is much better if you supply questions about your actual documents.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

GENERIC_IRRELEVANT = [
    "capital of Mongolia",
    "best pizza in Naples",
    "how do I change a car tire",
    "what is the boiling point of mercury",
    "who won the 1998 world cup",
    "recipe for sourdough starter",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relevant", action="append", default=[],
                    help="a question your vault SHOULD be able to answer")
    ap.add_argument("--irrelevant", action="append", default=[],
                    help="a question your vault should NOT answer")
    args = ap.parse_args()

    import ollama
    import main as vaultmind

    col = vaultmind.get_collection()
    if col.count() == 0:
        print("Vault is empty — index some documents first.")
        return 1

    relevant = args.relevant
    irrelevant = args.irrelevant or GENERIC_IRRELEVANT

    if not relevant:
        print("No --relevant questions given. Pass a few questions about documents\n"
              "you have actually indexed; without them this cannot find the gap.")
        return 1

    def nearest(question: str) -> float:
        emb = ollama.embeddings(
            model=vaultmind.EMBED_MODEL, prompt=question
        )["embedding"]
        res = col.query(query_embeddings=[emb], n_results=1, include=["distances"])
        return res["distances"][0][0]

    print(f"Vault: {col.count()} chunks, space={(col.metadata or {}).get('hnsw:space')}\n")

    hits = []
    print("Questions the vault should answer:")
    for q in relevant:
        d = nearest(q)
        hits.append(d)
        print(f"  {d:.3f}  {q}")

    misses = []
    print("\nQuestions it should not:")
    for q in irrelevant:
        d = nearest(q)
        misses.append(d)
        print(f"  {d:.3f}  {q}")

    worst_hit, best_miss = max(hits), min(misses)
    print(f"\nrelevant worst: {worst_hit:.3f}   irrelevant best: {best_miss:.3f}")

    if worst_hit >= best_miss:
        print(
            "\nNo clean separation — the two sets overlap, so no single threshold\n"
            "divides them. Usually this means the vault genuinely lacks answers to\n"
            "some 'relevant' questions, or the questions are too vague. Check which\n"
            "relevant question scored worst and confirm the document is indexed."
        )
        return 1

    suggested = round((worst_hit + best_miss) / 2, 2)
    current = vaultmind.relevance_threshold(col.count())

    print(f"\nSuggested VAULTMIND_RELEVANCE_THRESHOLD={suggested}")
    print(f"Currently using {current} for {col.count()} chunks.")

    if abs(suggested - current) < 0.03:
        print("Close enough — the built-in default fits this vault.")
    elif suggested > current:
        print(
            "\nThe default is stricter than your vault needs, so real answers\n"
            "may be getting dropped. Set the environment variable above."
        )
    else:
        print(
            "\nThe default is looser than your vault needs. That mostly costs\n"
            "a few tokens — the model is instructed to answer only from the\n"
            "sources and will say so when they do not support an answer — but\n"
            "you can tighten it with the environment variable above."
        )

    print(
        "\nNote: the built-in default scales with vault size, because the\n"
        "boundary moves as you index more. It is 0.58 under 50 chunks and\n"
        "0.45 over 500. Re-run this after your vault grows substantially."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

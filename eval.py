"""Eval harness: needle-in-haystack recall + answer quality across modes.

Reads needles.jsonl (produced by fetch_more.py), runs each question through
LightRAG in every mode (optionally with LLM reranker), checks whether the
retrieved context contains the gold fact and whether the answer contains
the expected substring.

Output: eval_results.json + console matrix.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from lightrag import QueryParam

from rag_core import build_rag, rerank_passages

NEEDLES_FILE = Path("./needles.jsonl")
RESULTS_FILE = Path("./eval_results.json")
MODES_DEFAULT = ["naive", "local", "global", "hybrid", "mix"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def contains(haystack: str, needle: str) -> bool:
    return norm(needle) in norm(haystack)


def load_needles() -> list[dict]:
    if not NEEDLES_FILE.exists():
        raise SystemExit(f"{NEEDLES_FILE} missing. Run fetch_more.py first.")
    return [json.loads(l) for l in NEEDLES_FILE.read_text().splitlines() if l.strip()]


def split_context(ctx: str) -> list[str]:
    """Rough split of LightRAG context string into passages for reranking."""
    chunks = re.split(r"\n{2,}|-----+|={3,}", ctx)
    return [c.strip() for c in chunks if len(c.strip()) > 80]


async def run_one(rag, needle: dict, mode: str, use_rerank: bool) -> dict:
    q = needle["question"]
    t0 = time.perf_counter()

    ctx = await rag.aquery(q, param=QueryParam(mode=mode, only_need_context=True))
    retrieval_hit = contains(ctx, needle["answer"]) or contains(ctx, needle["fact"])

    rerank_hit = None
    if use_rerank:
        passages = split_context(ctx)[:30]
        if passages:
            top = await rerank_passages(q, passages, top_k=5)
            rerank_hit = any(contains(p, needle["answer"]) for _, _, p in top)

    answer = await rag.aquery(q, param=QueryParam(mode=mode))
    answer_hit = contains(answer, needle["answer"])
    elapsed = time.perf_counter() - t0

    return {
        "needle_id": needle["id"],
        "mode": mode,
        "rerank": use_rerank,
        "retrieval_hit": retrieval_hit,
        "rerank_hit": rerank_hit,
        "answer_hit": answer_hit,
        "elapsed_s": round(elapsed, 2),
        "question": q,
        "expected": needle["answer"],
        "answer": answer[:400],
    }


def summarize(results: list[dict]) -> dict:
    summary: dict[tuple[str, bool], dict] = {}
    for r in results:
        key = (r["mode"], r["rerank"])
        s = summary.setdefault(key, {"n": 0, "retrieval": 0, "answer": 0, "rerank": 0, "rerank_n": 0, "time": 0.0})
        s["n"] += 1
        s["retrieval"] += int(r["retrieval_hit"])
        s["answer"] += int(r["answer_hit"])
        s["time"] += r["elapsed_s"]
        if r["rerank_hit"] is not None:
            s["rerank_n"] += 1
            s["rerank"] += int(r["rerank_hit"])
    return summary


def print_matrix(summary: dict) -> None:
    print("\n=== Eval Matrix ===")
    print(f"{'mode':<10}{'rerank':<8}{'retr@k':<10}{'rerank@5':<12}{'answer':<10}{'avg_s':<8}")
    for (mode, rr), s in sorted(summary.items()):
        r_retr = s["retrieval"] / s["n"]
        r_ans = s["answer"] / s["n"]
        r_rr = (s["rerank"] / s["rerank_n"]) if s["rerank_n"] else float("nan")
        avg = s["time"] / s["n"]
        print(f"{mode:<10}{str(rr):<8}{r_retr:<10.2%}{r_rr:<12.2%}{r_ans:<10.2%}{avg:<8.2f}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=MODES_DEFAULT)
    ap.add_argument("--rerank", action="store_true", help="Also eval with LLM reranker")
    ap.add_argument("--limit", type=int, default=0, help="Limit needles (0 = all)")
    args = ap.parse_args()

    needles = load_needles()
    if args.limit:
        needles = needles[: args.limit]

    rag = await build_rag()
    results: list[dict] = []
    configs = [(m, False) for m in args.modes]
    if args.rerank:
        configs += [(m, True) for m in args.modes]

    total = len(needles) * len(configs)
    i = 0
    for mode, rr in configs:
        for needle in needles:
            i += 1
            print(f"[{i}/{total}] mode={mode} rerank={rr} q={needle['question'][:60]}...")
            try:
                r = await run_one(rag, needle, mode, rr)
            except Exception as e:
                r = {"needle_id": needle["id"], "mode": mode, "rerank": rr,
                     "retrieval_hit": False, "rerank_hit": None, "answer_hit": False,
                     "elapsed_s": 0.0, "error": str(e),
                     "question": needle["question"], "expected": needle["answer"], "answer": ""}
            results.append(r)
            RESULTS_FILE.write_text(json.dumps(results, indent=2))

    print_matrix(summarize(results))
    print(f"\nFull results: {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())

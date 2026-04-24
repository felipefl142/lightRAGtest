"""Unit tests for eval.py pure helpers."""
from __future__ import annotations

import eval as ev


def test_norm_collapses_whitespace_and_lowers():
    assert ev.norm("  Hello   WORLD\n") == "hello world"


def test_contains_case_insensitive():
    assert ev.contains("The quick BROWN fox", "brown fox")
    assert not ev.contains("abc", "xyz")


def test_split_context_drops_short_fragments():
    ctx = "short\n\n" + ("long passage about widgets. " * 5) + "\n\n===\n\nmid" * 10
    parts = ev.split_context(ctx)
    assert all(len(p) > 80 for p in parts)
    assert any("widgets" in p for p in parts)


def test_summarize_aggregates_correctly():
    results = [
        {"mode": "hybrid", "rerank": False, "retrieval_hit": True,
         "rerank_hit": None, "answer_hit": True, "elapsed_s": 1.0},
        {"mode": "hybrid", "rerank": False, "retrieval_hit": False,
         "rerank_hit": None, "answer_hit": False, "elapsed_s": 3.0},
        {"mode": "hybrid", "rerank": True, "retrieval_hit": True,
         "rerank_hit": True, "answer_hit": True, "elapsed_s": 2.0},
    ]
    s = ev.summarize(results)
    assert s[("hybrid", False)]["n"] == 2
    assert s[("hybrid", False)]["retrieval"] == 1
    assert s[("hybrid", False)]["answer"] == 1
    assert s[("hybrid", False)]["time"] == 4.0
    assert s[("hybrid", True)]["rerank"] == 1
    assert s[("hybrid", True)]["rerank_n"] == 1

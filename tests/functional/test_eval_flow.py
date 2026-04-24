"""Functional: eval.run_one + full eval loop with fake rag."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

import eval as ev


NEEDLE = {
    "id": "abc123",
    "fact": "The azure falcon weighs 47.3 kilograms on Tuesday.",
    "question": "How much does the azure falcon weigh on Tuesday?",
    "answer": "47.3 kilograms",
    "file": "doc.txt",
    "position_ratio": 0.5,
}


@pytest.mark.functional
async def test_run_one_retrieval_and_answer_hit():
    rag = AsyncMock()
    rag.aquery = AsyncMock(side_effect=[
        "context snippet containing 47.3 kilograms of detail",   # ctx
        "The answer is 47.3 kilograms.",                          # final
    ])
    r = await ev.run_one(rag, NEEDLE, mode="hybrid", use_rerank=False)
    assert r["retrieval_hit"] is True
    assert r["answer_hit"] is True
    assert r["rerank_hit"] is None
    assert r["mode"] == "hybrid"


@pytest.mark.functional
async def test_run_one_miss_when_no_overlap():
    rag = AsyncMock()
    rag.aquery = AsyncMock(side_effect=["unrelated text", "dunno"])
    r = await ev.run_one(rag, NEEDLE, mode="naive", use_rerank=False)
    assert r["retrieval_hit"] is False
    assert r["answer_hit"] is False


@pytest.mark.functional
async def test_run_one_rerank_path(monkeypatch):
    rag = AsyncMock()
    long = "x" * 200
    hit = "gold passage that mentions 47.3 kilograms of weight evidence here " * 3
    ctx = f"{long}\n\n{hit}\n\n{long}"
    rag.aquery = AsyncMock(side_effect=[ctx, "answer: 47.3 kilograms"])

    async def fake_rerank(q, passages, top_k=5, model=None):
        return [(i, 10.0 - i, p) for i, p in enumerate(passages[:top_k])]

    monkeypatch.setattr(ev, "rerank_passages", fake_rerank)
    r = await ev.run_one(rag, NEEDLE, mode="mix", use_rerank=True)
    assert r["rerank_hit"] is True


@pytest.mark.functional
def test_load_needles_missing_exits(tmp_cwd, monkeypatch):
    monkeypatch.setattr(ev, "NEEDLES_FILE", tmp_cwd / "no.jsonl")
    with pytest.raises(SystemExit):
        ev.load_needles()


@pytest.mark.functional
def test_load_needles_parses(tmp_cwd, monkeypatch):
    f = tmp_cwd / "n.jsonl"
    f.write_text(json.dumps(NEEDLE) + "\n")
    monkeypatch.setattr(ev, "NEEDLES_FILE", f)
    out = ev.load_needles()
    assert out[0]["id"] == "abc123"

"""Integration: rerank_passages against stubbed ollama client."""
from __future__ import annotations

import pytest

import rag_core


@pytest.mark.integration
async def test_rerank_sorts_by_score(monkeypatch):
    scores = iter(["2", "9", "5"])

    class Client:
        def __init__(self, host=None):
            pass

        async def generate(self, model, prompt, options=None):
            return {"response": next(scores)}

    import ollama
    monkeypatch.setattr(ollama, "AsyncClient", Client)

    passages = ["low", "high", "mid"]
    out = await rag_core.rerank_passages("q", passages, top_k=3)
    assert [p for _, _, p in out] == ["high", "mid", "low"]
    assert [s for _, s, _ in out] == [9.0, 5.0, 2.0]


@pytest.mark.integration
async def test_rerank_handles_bad_response(monkeypatch):
    class Client:
        def __init__(self, host=None):
            pass

        async def generate(self, model, prompt, options=None):
            return {"response": "no digit here"}

    import ollama
    monkeypatch.setattr(ollama, "AsyncClient", Client)

    out = await rag_core.rerank_passages("q", ["a", "b"], top_k=2)
    assert all(s == 0.0 for _, s, _ in out)


@pytest.mark.integration
async def test_rerank_truncates_to_top_k(monkeypatch):
    class Client:
        def __init__(self, host=None):
            pass

        async def generate(self, model, prompt, options=None):
            return {"response": "5"}

    import ollama
    monkeypatch.setattr(ollama, "AsyncClient", Client)

    out = await rag_core.rerank_passages("q", ["a", "b", "c", "d"], top_k=2)
    assert len(out) == 2

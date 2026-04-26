"""Shared fixtures. Stubs ollama + lightrag so tests never touch network/GPU."""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch):
    """Run test with cwd = tmp_path so ledger/needles files land in a sandbox."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def fake_ollama(monkeypatch):
    """Patch ollama.AsyncClient. Returns sent prompts + controllable response."""
    calls: list[dict] = []
    state = {"response": "7"}

    class FakeClient:
        def __init__(self, host: str | None = None):
            self.host = host

        async def generate(self, model, prompt, options=None):
            calls.append({"model": model, "prompt": prompt, "options": options})
            return {"response": state["response"]}

    import ollama
    monkeypatch.setattr(ollama, "AsyncClient", FakeClient)
    return {"calls": calls, "state": state}


@pytest.fixture
def fake_rag():
    """Return an AsyncMock shaped like LightRAG used by ingest/eval."""
    rag = MagicMock()
    rag.ainsert = AsyncMock()
    rag.adelete_by_doc_id = AsyncMock()
    rag.aquery = AsyncMock(return_value="stub answer")
    rag.initialize_storages = AsyncMock()
    return rag

"""Functional: ingest.main end-to-end with fake rag + tmp corpus."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import ingest


@pytest.mark.functional
def test_ingest_skips_already_ingested(tmp_path, monkeypatch, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello")
    ledger = tmp_path / "led.json"
    monkeypatch.setattr(ingest, "DOCS_DIR", docs)
    monkeypatch.setattr(ingest, "LEDGER", ledger)

    fake = AsyncMock()
    fake.ainsert = AsyncMock()
    monkeypatch.setattr(ingest, "build_rag", AsyncMock(return_value=fake))

    asyncio.run(ingest.main())
    assert fake.ainsert.await_count == 1

    asyncio.run(ingest.main())
    assert fake.ainsert.await_count == 1  # unchanged


@pytest.mark.functional
def test_ingest_reingests_when_content_changes(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    f = docs / "a.txt"
    f.write_text("v1")
    monkeypatch.setattr(ingest, "DOCS_DIR", docs)
    monkeypatch.setattr(ingest, "LEDGER", tmp_path / "led.json")

    fake = AsyncMock()
    fake.ainsert = AsyncMock()
    monkeypatch.setattr(ingest, "build_rag", AsyncMock(return_value=fake))

    asyncio.run(ingest.main())
    f.write_text("v2 bigger payload")
    asyncio.run(ingest.main())
    assert fake.ainsert.await_count == 2


@pytest.mark.functional
def test_ingest_exits_if_docs_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "DOCS_DIR", tmp_path / "nope")
    with pytest.raises(SystemExit):
        asyncio.run(ingest.main())

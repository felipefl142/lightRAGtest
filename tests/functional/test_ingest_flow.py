"""Functional: ingest.main end-to-end with fake rag + tmp corpus."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import ingest


def _argv(docs: str, *extra: str) -> list[str]:
    return ["--docs-dir", docs, *extra]


@pytest.mark.functional
def test_ingest_skips_already_ingested(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello")
    monkeypatch.setattr(ingest, "LEDGER", tmp_path / "led.json")

    fake = AsyncMock()
    fake.ainsert = AsyncMock()
    fake.adelete_by_doc_id = AsyncMock()
    monkeypatch.setattr(ingest, "build_rag", AsyncMock(return_value=fake))

    asyncio.run(ingest.main(_argv(str(docs))))
    assert fake.ainsert.await_count == 1

    asyncio.run(ingest.main(_argv(str(docs))))
    assert fake.ainsert.await_count == 1


@pytest.mark.functional
def test_ingest_reingests_when_content_changes(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    docs = tmp_path / "docs"
    docs.mkdir()
    f = docs / "a.txt"
    f.write_text("v1")
    monkeypatch.setattr(ingest, "LEDGER", tmp_path / "led.json")

    manager = MagicMock()
    fake = AsyncMock()
    fake.ainsert = AsyncMock()
    fake.adelete_by_doc_id = AsyncMock()
    manager.attach_mock(fake.ainsert, "ainsert")
    manager.attach_mock(fake.adelete_by_doc_id, "adelete_by_doc_id")
    monkeypatch.setattr(ingest, "build_rag", AsyncMock(return_value=fake))

    asyncio.run(ingest.main(_argv(str(docs))))
    f.write_text("v2 bigger payload")
    asyncio.run(ingest.main(_argv(str(docs))))

    assert fake.ainsert.await_count == 2
    assert fake.adelete_by_doc_id.await_count == 1
    fake.adelete_by_doc_id.assert_awaited_with("a")

    names = [c[0] for c in manager.mock_calls]
    delete_idx = names.index("adelete_by_doc_id")
    second_insert_idx = [i for i, n in enumerate(names) if n == "ainsert"][1]
    assert delete_idx < second_insert_idx, (
        f"delete must come before re-insert: {names}"
    )


@pytest.mark.functional
def test_ingest_purges_removed_files(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    f = docs / "a.txt"
    f.write_text("hello")
    monkeypatch.setattr(ingest, "LEDGER", tmp_path / "led.json")

    fake = AsyncMock()
    fake.ainsert = AsyncMock()
    fake.adelete_by_doc_id = AsyncMock()
    monkeypatch.setattr(ingest, "build_rag", AsyncMock(return_value=fake))

    asyncio.run(ingest.main(_argv(str(docs))))
    assert fake.ainsert.await_count == 1

    f.unlink()
    asyncio.run(ingest.main(_argv(str(docs))))

    assert fake.adelete_by_doc_id.await_count == 1
    fake.adelete_by_doc_id.assert_awaited_with("a")
    assert ingest.load_ledger() == {}


@pytest.mark.functional
def test_ingest_dry_run_does_not_touch_rag(tmp_path, monkeypatch, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello")
    monkeypatch.setattr(ingest, "LEDGER", tmp_path / "led.json")

    build_rag = AsyncMock()
    monkeypatch.setattr(ingest, "build_rag", build_rag)

    asyncio.run(ingest.main(_argv(str(docs), "--dry-run")))

    out = capsys.readouterr().out
    assert "add:    1" in out
    assert "(no changes made)" in out
    build_rag.assert_not_awaited()
    assert not (tmp_path / "led.json").exists()


@pytest.mark.functional
def test_ingest_pattern_flag_filters_files(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("text")
    (docs / "b.md").write_text("md")
    monkeypatch.setattr(ingest, "LEDGER", tmp_path / "led.json")

    fake = AsyncMock()
    fake.ainsert = AsyncMock()
    fake.adelete_by_doc_id = AsyncMock()
    monkeypatch.setattr(ingest, "build_rag", AsyncMock(return_value=fake))

    asyncio.run(ingest.main(_argv(str(docs), "--pattern", "*.md")))
    assert fake.ainsert.await_count == 1
    fake.ainsert.assert_awaited_with("md", ids=["b"], file_paths=["b.md"])


@pytest.mark.functional
def test_ingest_concurrency_runs_inserts_in_parallel(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(4):
        (docs / f"f{i}.txt").write_text(f"v{i}")
    monkeypatch.setattr(ingest, "LEDGER", tmp_path / "led.json")

    in_flight = {"now": 0, "max": 0}

    fake = AsyncMock()

    async def slow_insert(*args, **kwargs):
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        await asyncio.sleep(0.05)
        in_flight["now"] -= 1

    fake.ainsert = AsyncMock(side_effect=slow_insert)
    fake.adelete_by_doc_id = AsyncMock()
    monkeypatch.setattr(ingest, "build_rag", AsyncMock(return_value=fake))

    asyncio.run(ingest.main(_argv(str(docs), "--concurrency", "3")))

    assert fake.ainsert.await_count == 4
    assert in_flight["max"] >= 3  # all permitted slots in use


@pytest.mark.functional
def test_ingest_exits_if_docs_dir_missing(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(SystemExit):
        asyncio.run(ingest.main(_argv(str(missing))))

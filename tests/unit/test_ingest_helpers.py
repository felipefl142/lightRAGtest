"""Unit tests for ingest.py pure helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ingest


def test_digest_matches_sha256(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello")
    assert ingest.digest(f) == hashlib.sha256(b"hello").hexdigest()


def test_load_ledger_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "LEDGER", tmp_path / "nope.json")
    assert ingest.load_ledger() == {}


def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr(ingest, "LEDGER", ledger)
    ingest.save_ledger({"a.txt": "deadbeef"})
    assert json.loads(ledger.read_text()) == {"a.txt": "deadbeef"}
    assert ingest.load_ledger() == {"a.txt": "deadbeef"}


def test_diff_files_classifies_each_category():
    disk =   {"a.txt": "h1", "b.txt": "h2_new", "c.txt": "h3"}
    ledger = {"a.txt": "h1", "b.txt": "h2_old", "d.txt": "h4"}
    diff = ingest.diff_files(disk, ledger)
    assert diff.new == ["c.txt"]
    assert diff.changed == ["b.txt"]
    assert diff.removed == ["d.txt"]
    assert diff.unchanged == ["a.txt"]


def test_diff_files_sorted_output():
    disk =   {"z.txt": "h", "a.txt": "h"}
    ledger = {}
    diff = ingest.diff_files(disk, ledger)
    assert diff.new == ["a.txt", "z.txt"]


def test_diff_files_empty_inputs():
    diff = ingest.diff_files({}, {})
    assert diff.new == diff.changed == diff.removed == diff.unchanged == []


def test_format_plan_counts_only_when_large():
    diff = ingest.Diff(
        new=[f"n{i}.txt" for i in range(15)],
        changed=[f"c{i}.txt" for i in range(10)],
        removed=[],
        unchanged=[],
    )
    out = ingest.format_plan(diff)
    assert "add:    15" in out
    assert "update: 10" in out
    assert "delete: 0" in out
    assert "skip:   0" in out
    assert "n0.txt" not in out  # too many to list


def test_format_plan_lists_files_when_small():
    diff = ingest.Diff(
        new=["a.txt"], changed=["b.txt"], removed=["c.txt"], unchanged=["d.txt"]
    )
    out = ingest.format_plan(diff)
    assert "add:    1" in out
    assert "  + a.txt" in out
    assert "  ~ b.txt" in out
    assert "  - c.txt" in out
    assert "(no changes made)" in out


def test_parse_args_defaults(monkeypatch):
    monkeypatch.delenv("SAMPLE_DOCS_DIR", raising=False)
    args = ingest.parse_args([])
    assert str(args.docs_dir) == "sample_docs"
    assert args.pattern == "*.txt"
    assert args.dry_run is False
    assert args.concurrency == 1


def test_parse_args_env_override_for_docs_dir(monkeypatch):
    monkeypatch.setenv("SAMPLE_DOCS_DIR", "/tmp/elsewhere")
    args = ingest.parse_args([])
    assert str(args.docs_dir) == "/tmp/elsewhere"


def test_parse_args_explicit_flags(monkeypatch):
    monkeypatch.delenv("SAMPLE_DOCS_DIR", raising=False)
    args = ingest.parse_args([
        "--docs-dir", "/x",
        "--pattern", "**/*.md",
        "--dry-run",
        "--concurrency", "4",
    ])
    assert str(args.docs_dir) == "/x"
    assert args.pattern == "**/*.md"
    assert args.dry_run is True
    assert args.concurrency == 4


def test_parse_args_concurrency_must_be_positive():
    import pytest
    with pytest.raises(SystemExit):
        ingest.parse_args(["--concurrency", "0"])

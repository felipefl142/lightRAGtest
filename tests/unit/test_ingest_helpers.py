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

"""Integration: needle injection writes manifest + mutates docs."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import fetch_more as fm


@pytest.mark.integration
def test_inject_needles_writes_manifest_and_modifies_docs(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "NEEDLES_FILE", tmp_path / "needles.jsonl")
    files = []
    for i in range(3):
        p = tmp_path / f"doc_{i}.txt"
        p.write_text("para1\n\n" + ("filler paragraph. " * 200) + "\n\nparaN")
        files.append(p)

    random.seed(0)
    needles = fm.inject_needles(files, count=5)
    assert len(needles) == 5
    manifest = [json.loads(l) for l in (tmp_path / "needles.jsonl").read_text().splitlines()]
    assert len(manifest) == 5
    for n in needles:
        target = tmp_path / n["file"]
        assert n["fact"] in target.read_text()


@pytest.mark.integration
def test_inject_needles_skips_if_manifest_exists(tmp_path, monkeypatch):
    needles_file = tmp_path / "needles.jsonl"
    needles_file.write_text(json.dumps({"id": "x", "fact": "f", "question": "q",
                                         "answer": "a", "file": "f.txt",
                                         "position_ratio": 0.5}))
    monkeypatch.setattr(fm, "NEEDLES_FILE", needles_file)
    out = fm.inject_needles([], count=5)
    assert len(out) == 1

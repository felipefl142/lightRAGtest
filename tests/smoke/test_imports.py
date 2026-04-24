"""Smoke: modules import, env defaults sane, expected files exist."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.smoke
def test_all_modules_import():
    import rag_core
    import ingest
    import eval as ev
    import fetch_more
    import fetch_data
    import app  # noqa: F401 — streamlit import is heavy but must not error


@pytest.mark.smoke
def test_rag_core_defaults_reasonable():
    import rag_core as rc
    assert rc.EMBED_DIM > 0
    assert rc.LLM_NUM_CTX >= 1024
    assert rc.CHUNK_TOKEN_SIZE > 0
    assert rc.LLM_TIMEOUT >= 60
    assert rc.LLM_MAX_ASYNC >= 1


@pytest.mark.smoke
def test_expected_project_files_present():
    for name in ["rag_core.py", "ingest.py", "eval.py", "fetch_more.py",
                 "fetch_data.py", "app.py", "pyproject.toml"]:
        assert (ROOT / name).exists(), f"missing {name}"


@pytest.mark.smoke
def test_needle_templates_nonempty():
    import fetch_more as fm
    assert len(fm.NEEDLE_TEMPLATES) >= 1
    assert all(fm.SLOTS.values())

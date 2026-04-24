"""Integration: wiki_get retry/backoff + fetch_wiki parse. No real HTTP."""
from __future__ import annotations

import pytest
import responses

import fetch_more as fm


@pytest.mark.integration
@responses.activate
def test_wiki_get_retries_on_429(monkeypatch):
    monkeypatch.setattr(fm, "REQ_DELAY", 0)
    monkeypatch.setattr("time.sleep", lambda *_: None)  # skip backoff sleep
    responses.add(responses.GET, fm.WIKI_API, status=429, headers={"Retry-After": "0"})
    responses.add(responses.GET, fm.WIKI_API, json={"ok": True}, status=200)
    assert fm.wiki_get({"action": "x"}) == {"ok": True}


@pytest.mark.integration
@responses.activate
def test_wiki_get_gives_up_after_max(monkeypatch):
    monkeypatch.setattr(fm, "REQ_DELAY", 0)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    for _ in range(fm.MAX_RETRIES):
        responses.add(responses.GET, fm.WIKI_API, status=429)
    with pytest.raises(RuntimeError):
        fm.wiki_get({"action": "x"})


@pytest.mark.integration
@responses.activate
def test_fetch_wiki_returns_prefixed_markdown(monkeypatch):
    monkeypatch.setattr(fm, "REQ_DELAY", 0)
    responses.add(
        responses.GET,
        fm.WIKI_API,
        json={"query": {"pages": {"1": {"extract": "Body text here."}}}},
        status=200,
    )
    out = fm.fetch_wiki("Some_Title")
    assert out.startswith("# Some Title")
    assert "Body text here." in out


@pytest.mark.integration
@responses.activate
def test_fetch_wiki_missing_returns_empty(monkeypatch):
    monkeypatch.setattr(fm, "REQ_DELAY", 0)
    responses.add(
        responses.GET,
        fm.WIKI_API,
        json={"query": {"pages": {"-1": {"missing": ""}}}},
        status=200,
    )
    assert fm.fetch_wiki("Nonexistent") == ""

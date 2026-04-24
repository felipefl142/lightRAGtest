"""Unit tests for pure helpers in fetch_more.py."""
from __future__ import annotations

import random
import re

import fetch_more as fm


def test_sanitize_lowercases_and_strips():
    assert fm.sanitize("Foo Bar/Baz") == "wiki_foo_bar_baz.txt"


def test_sanitize_handles_unicode_and_punctuation():
    assert fm.sanitize("Béatrix D'ussane") == "wiki_b_atrix_d_ussane.txt"


def test_fill_substitutes_all_placeholders():
    random.seed(0)
    out, picks = fm.fill("The {animal} in {place} weighs {n}kg")
    assert "{" not in out
    assert picks["animal"] in fm.SLOTS["animal"]
    assert picks["place"] in fm.SLOTS["place"]
    assert picks["n"] in fm.SLOTS["n"]


def test_fill_same_key_gets_same_value():
    random.seed(1)
    out, picks = fm.fill("{animal} and {animal} again")
    parts = out.split(" and ")
    assert parts[0] == parts[1].replace(" again", "")


def test_needle_templates_consistent():
    """Every template's q + a must reference only keys present in SLOTS."""
    allowed = set(fm.SLOTS)
    for fact, q, a in fm.NEEDLE_TEMPLATES:
        for text in (fact, q, a):
            for key in re.findall(r"\{(\w+)\}", text):
                assert key in allowed, f"unknown slot {key}"

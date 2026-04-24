"""Expand corpus to ~100 Wikipedia docs + inject synthetic needles.

- Pulls random Wikipedia articles via MediaWiki API until target count reached.
- Injects N unique needle facts into random docs at random positions.
- Writes needles.jsonl manifest for eval harness.

Re-run safe: skips files already present, appends new needles only if
needles.jsonl missing.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

OUT_DIR = Path(os.getenv("SAMPLE_DOCS_DIR", "./sample_docs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
NEEDLES_FILE = Path("./needles.jsonl")

TARGET_DOCS = int(os.getenv("TARGET_DOCS", "100"))
NEEDLE_COUNT = int(os.getenv("NEEDLE_COUNT", "15"))
MIN_CHARS = 2000
UA = {"User-Agent": "lightRAGtest/0.1 (learning project; contact: felipefrl142@gmail.com)"}
WIKI_API = "https://en.wikipedia.org/w/api.php"
REQ_DELAY = float(os.getenv("REQ_DELAY", "0.3"))
MAX_RETRIES = 5


def wiki_get(params: dict) -> dict:
    """GET with exponential backoff on 429/5xx."""
    for attempt in range(MAX_RETRIES):
        r = requests.get(WIKI_API, params=params, headers=UA, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            wait = (2 ** attempt) + random.random()
            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = max(wait, int(retry_after))
            print(f"  rate-limited ({r.status_code}), sleep {wait:.1f}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        time.sleep(REQ_DELAY)
        return r.json()
    raise RuntimeError(f"Giving up after {MAX_RETRIES} retries")

NEEDLE_TEMPLATES = [
    ("The {animal} of {place} weighs exactly {n} kilograms on {day}.",
     "How much does the {animal} of {place} weigh on {day}?",
     "{n} kilograms"),
    ("Project {code} was commissioned in {year} by the {org} council.",
     "In what year was Project {code} commissioned?",
     "{year}"),
    ("The secret passphrase for vault {code} is '{phrase}'.",
     "What is the secret passphrase for vault {code}?",
     "{phrase}"),
    ("{person} invented the {gadget} in the year {year}.",
     "Who invented the {gadget}?",
     "{person}"),
    ("The {color} {gem} of {place} was discovered in {year}.",
     "When was the {color} {gem} of {place} discovered?",
     "{year}"),
]

SLOTS = {
    "animal": ["azure falcon", "copper wombat", "glass otter", "tin lynx", "velvet stoat"],
    "place": ["Zanzibar", "Ulaanbaatar", "Reykjavik", "Qaraghandy", "Bujumbura"],
    "day": ["Tuesday", "Friday", "Sunday"],
    "code": ["QZ-417", "MX-993", "RT-158", "AL-602", "KP-274"],
    "org": ["Helvellyn", "Brindlewood", "Karakoram", "Fjordhavn"],
    "phrase": ["thunder-pickle-9", "violet-mammoth-7", "quiet-glacier-3", "bronze-otter-11"],
    "person": ["Ingrid Larsson-Mak", "Tomasz Wendigo", "Yui Okonkwo-Brahms"],
    "gadget": ["hyperbaric teapot", "quantum doorknob", "tessellated umbrella"],
    "color": ["mauve", "ochre", "cerulean"],
    "gem": ["peridot", "sphene", "tanzanite"],
    "year": ["1847", "1923", "1956", "1991", "2012"],
    "n": ["47.3", "112.7", "8.4", "231.9", "66.6"],
}


def fill(template: str) -> tuple[str, dict]:
    picks: dict[str, str] = {}
    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in picks:
            picks[key] = random.choice(SLOTS[key])
        return picks[key]
    return re.sub(r"\{(\w+)\}", sub, template), picks


def random_wiki_titles(n: int) -> list[str]:
    """Grab n random article titles via MediaWiki random endpoint."""
    titles: list[str] = []
    while len(titles) < n:
        batch = min(20, n - len(titles))
        data = wiki_get({
            "action": "query",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": batch,
            "format": "json",
        })
        for item in data["query"]["random"]:
            titles.append(item["title"].replace(" ", "_"))
    return titles


def fetch_wiki(title: str) -> str:
    data = wiki_get({
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
        "redirects": 1,
    })
    page = next(iter(data["query"]["pages"].values()))
    if "missing" in page:
        return ""
    extract = page.get("extract", "").strip()
    return f"# {title.replace('_', ' ')}\n\n{extract}" if extract else ""


def sanitize(title: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_").lower()
    return f"wiki_{s}.txt"


def grow_corpus() -> list[Path]:
    """Fetch until OUT_DIR has TARGET_DOCS files."""
    have = sorted(OUT_DIR.glob("*.txt"))
    need = TARGET_DOCS - len(have)
    if need <= 0:
        print(f"Corpus already has {len(have)} docs. Skipping fetch.")
        return have

    print(f"Need {need} more docs (have {len(have)}, target {TARGET_DOCS}).")
    added = 0
    while added < need:
        titles = random_wiki_titles(min(30, (need - added) * 2))
        for title in titles:
            if added >= need:
                break
            fname = sanitize(title)
            path = OUT_DIR / fname
            if path.exists():
                continue
            text = fetch_wiki(title)
            if not text or len(text) < MIN_CHARS:
                continue
            path.write_text(text, encoding="utf-8")
            added += 1
            print(f"  [{added}/{need}] {fname} ({len(text)} chars)")
    return sorted(OUT_DIR.glob("*.txt"))


def inject_needles(files: list[Path], count: int) -> list[dict]:
    if NEEDLES_FILE.exists():
        existing = [json.loads(l) for l in NEEDLES_FILE.read_text().splitlines() if l.strip()]
        print(f"Needles already injected ({len(existing)}). Skipping.")
        return existing

    needles: list[dict] = []
    random.seed(42)
    pool = [f for f in files if f.stat().st_size > MIN_CHARS]
    for _ in range(count):
        tpl_fact, tpl_q, tpl_a = random.choice(NEEDLE_TEMPLATES)
        fact, picks = fill(tpl_fact)
        question = re.sub(r"\{(\w+)\}", lambda m: picks[m.group(1)], tpl_q)
        answer = re.sub(r"\{(\w+)\}", lambda m: picks[m.group(1)], tpl_a)

        target = random.choice(pool)
        text = target.read_text(encoding="utf-8")
        paras = text.split("\n\n")
        pos = random.randint(1, max(1, len(paras) - 1))
        paras.insert(pos, fact)
        target.write_text("\n\n".join(paras), encoding="utf-8")

        needles.append({
            "id": str(uuid.uuid4())[:8],
            "fact": fact,
            "question": question,
            "answer": answer,
            "file": target.name,
            "position_ratio": round(pos / max(1, len(paras)), 3),
        })
        print(f"  needle -> {target.name}: {fact[:60]}...")

    NEEDLES_FILE.write_text("\n".join(json.dumps(n) for n in needles))
    print(f"Wrote {len(needles)} needles to {NEEDLES_FILE}")
    return needles


def main() -> None:
    files = grow_corpus()
    inject_needles(files, NEEDLE_COUNT)


if __name__ == "__main__":
    main()

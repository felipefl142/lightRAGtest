"""Download small public-domain corpus into sample_docs/.

Sources: two Paul Graham essays + five Wikipedia articles related to RAG.
Re-running skips files already present.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

OUT_DIR = Path(os.getenv("SAMPLE_DOCS_DIR", "./sample_docs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PG_ESSAYS = {
    "pg_great_work.txt": "https://www.paulgraham.com/greatwork.html",
    "pg_makers_schedule.txt": "https://www.paulgraham.com/makersschedule.html",
}

WIKI_TITLES = [
    "Retrieval-augmented_generation",
    "Knowledge_graph",
    "Large_language_model",
    "Vector_database",
    "Graph_database",
]

UA = {"User-Agent": "lightRAGtest/0.1 (learning project)"}


def fetch_pg(url: str) -> str:
    html = requests.get(url, headers=UA, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def fetch_wiki(title: str) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params=params,
        headers=UA,
        timeout=30,
    )
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        return ""
    extract = page.get("extract", "").strip()
    return f"# {title.replace('_', ' ')}\n\n{extract}" if extract else ""


def write_if_missing(path: Path, content: str) -> bool:
    if path.exists():
        print(f"skip {path.name} (exists)")
        return False
    path.write_text(content, encoding="utf-8")
    print(f"wrote {path.name} ({len(content)} chars)")
    return True


def main() -> None:
    for fname, url in PG_ESSAYS.items():
        write_if_missing(OUT_DIR / fname, fetch_pg(url))
    for title in WIKI_TITLES:
        fname = f"wiki_{title.lower()}.txt"
        text = fetch_wiki(title)
        if not text:
            print(f"WARN: no wiki page for {title}, skipping")
            continue
        write_if_missing(OUT_DIR / fname, text)


if __name__ == "__main__":
    main()

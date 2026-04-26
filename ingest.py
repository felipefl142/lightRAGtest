"""Insert every *.txt in sample_docs/ into LightRAG.

Tracks inserted files via .ingested.json in the working dir so re-runs
skip docs already indexed. Delete that file (or the whole rag_storage/
dir) to force a reindex.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from rag_core import WORKING_DIR, build_rag

load_dotenv()

DOCS_DIR = Path(os.getenv("SAMPLE_DOCS_DIR", "./sample_docs"))
LEDGER = Path(WORKING_DIR) / ".ingested.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ledger() -> dict[str, str]:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {}


def save_ledger(data: dict[str, str]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(data, indent=2))


@dataclass(frozen=True)
class Diff:
    new: list[str]
    changed: list[str]
    removed: list[str]
    unchanged: list[str]


def diff_files(disk: dict[str, str], ledger: dict[str, str]) -> Diff:
    disk_keys = set(disk)
    ledger_keys = set(ledger)
    common = disk_keys & ledger_keys
    return Diff(
        new=sorted(disk_keys - ledger_keys),
        changed=sorted(k for k in common if disk[k] != ledger[k]),
        removed=sorted(ledger_keys - disk_keys),
        unchanged=sorted(k for k in common if disk[k] == ledger[k]),
    )


def format_plan(diff: Diff) -> str:
    total_changes = len(diff.new) + len(diff.changed) + len(diff.removed)
    lines = [
        "plan:",
        f"  add:    {len(diff.new)} files",
        f"  update: {len(diff.changed)} files",
        f"  delete: {len(diff.removed)} files",
        f"  skip:   {len(diff.unchanged)} files",
    ]
    if total_changes <= 20:
        for name in diff.new:
            lines.append(f"  + {name}")
        for name in diff.changed:
            lines.append(f"  ~ {name}")
        for name in diff.removed:
            lines.append(f"  - {name}")
    lines.append("(no changes made)")
    return "\n".join(lines)


async def main() -> None:
    if not DOCS_DIR.exists():
        raise SystemExit(f"{DOCS_DIR} missing. Run `python fetch_data.py` first.")

    files = sorted(DOCS_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"No *.txt under {DOCS_DIR}.")

    ledger = load_ledger()
    pending: list[tuple[Path, str]] = []
    for f in files:
        h = digest(f)
        if ledger.get(f.name) == h:
            print(f"skip {f.name} (already ingested)")
            continue
        pending.append((f, h))

    if not pending:
        print("Nothing new to ingest.")
        return

    total = len(pending)
    skipped = len(files) - total
    print(f"{total} pending, {skipped} skipped, {len(files)} total")

    rag = await build_rag()
    for i, (path, h) in enumerate(pending, start=1):
        text = path.read_text(encoding="utf-8")
        print(f"[{i}/{total}] ingest {path.name} ({len(text)} chars)...", flush=True)
        await rag.ainsert(text, ids=[path.stem], file_paths=[path.name])
        ledger[path.name] = h
        save_ledger(ledger)
        print(f"[{i}/{total}] done {path.name}", flush=True)
    print(f"Done. {total}/{total} ingested.")


if __name__ == "__main__":
    asyncio.run(main())

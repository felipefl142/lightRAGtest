"""Resync sample_docs/ into LightRAG.

Tracks ingested files via .ingested.json keyed by POSIX relpath. On each
run, computes a diff: new files are inserted, changed files are
delete-then-inserted, removed files are deleted from the KG. The ledger
is updated under an asyncio.Lock so partial progress survives crashes.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from rag_core import WORKING_DIR, build_rag

load_dotenv()

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


def _positive_int(raw: str) -> int:
    n = int(raw)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resync sample_docs into LightRAG.")
    p.add_argument(
        "--docs-dir",
        type=Path,
        default=Path(os.getenv("SAMPLE_DOCS_DIR", "sample_docs")),
    )
    p.add_argument("--pattern", default="*.txt")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--concurrency", type=_positive_int, default=1)
    return p.parse_args(argv)


def scan_disk(docs_dir: Path, pattern: str) -> dict[str, str]:
    """Return {posix_relpath: sha256} for files matching pattern."""
    out: dict[str, str] = {}
    for f in sorted(docs_dir.glob(pattern)):
        if not f.is_file():
            continue
        rel = f.relative_to(docs_dir).as_posix()
        out[rel] = digest(f)
    return out


async def _insert_file(rag, docs_dir: Path, rel: str) -> None:
    path = docs_dir / rel
    text = path.read_text(encoding="utf-8")
    doc_id = Path(rel).stem
    await rag.ainsert(text, ids=[doc_id], file_paths=[rel])


async def _delete_file(rag, rel: str) -> None:
    await rag.adelete_by_doc_id(Path(rel).stem)


async def _apply(
    rag,
    docs_dir: Path,
    diff: Diff,
    disk: dict[str, str],
    ledger: dict[str, str],
    concurrency: int,
) -> None:
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    total = len(diff.new) + len(diff.changed) + len(diff.removed)
    counter = {"done": 0}

    async def run_new(rel: str) -> None:
        async with sem:
            await _insert_file(rag, docs_dir, rel)
            async with lock:
                ledger[rel] = disk[rel]
                save_ledger(ledger)
                counter["done"] += 1
                print(f"[{counter['done']}/{total}] + {rel}", flush=True)

    async def run_changed(rel: str) -> None:
        async with sem:
            await _delete_file(rag, rel)
            await _insert_file(rag, docs_dir, rel)
            async with lock:
                ledger[rel] = disk[rel]
                save_ledger(ledger)
                counter["done"] += 1
                print(f"[{counter['done']}/{total}] ~ {rel}", flush=True)

    async def run_removed(rel: str) -> None:
        async with sem:
            await _delete_file(rag, rel)
            async with lock:
                ledger.pop(rel, None)
                save_ledger(ledger)
                counter["done"] += 1
                print(f"[{counter['done']}/{total}] - {rel}", flush=True)

    tasks = (
        [run_new(r) for r in diff.new]
        + [run_changed(r) for r in diff.changed]
        + [run_removed(r) for r in diff.removed]
    )
    if tasks:
        await asyncio.gather(*tasks)


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    docs_dir: Path = args.docs_dir

    if not docs_dir.exists():
        raise SystemExit(f"{docs_dir} missing. Run `python fetch_data.py` first.")

    disk = scan_disk(docs_dir, args.pattern)
    ledger = load_ledger()
    diff = diff_files(disk, ledger)

    if args.dry_run:
        print(format_plan(diff))
        return

    if not (diff.new or diff.changed or diff.removed):
        print("nothing to ingest")
        return

    print(format_plan(diff).replace("(no changes made)", "").rstrip())
    rag = await build_rag()
    await _apply(rag, docs_dir, diff, disk, ledger, args.concurrency)
    print("done.")


if __name__ == "__main__":
    asyncio.run(main())

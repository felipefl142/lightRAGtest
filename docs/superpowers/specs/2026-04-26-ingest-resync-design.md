# Ingest Resync — Design

Date: 2026-04-26
Status: approved

## Goal

Make `ingest.py` re-ingest changed files, ingest new files, and purge removed
files from the LightRAG store. Add CLI flags for docs dir, glob pattern,
dry-run, and concurrency.

## Background

Current `ingest.py` skips files whose SHA-256 matches the ledger entry but
otherwise calls `rag.ainsert` again. LightRAG's KG retains stale entities
unless `adelete_by_doc_id` is called first (confirmed by `reingest_two.py`).
Files removed from disk leave ghost docs in the store. Behavior is hardcoded
to `*.txt` under `SAMPLE_DOCS_DIR` and runs sequentially.

## Scope

In:

- New file → insert
- Changed file (hash mismatch) → delete-by-doc-id, then insert
- Removed file (in ledger, gone from disk) → delete-by-doc-id, drop ledger entry
- CLI flags: `--docs-dir`, `--pattern`, `--dry-run`, `--concurrency`
- Concurrency via `asyncio.Semaphore`, ledger writes guarded by `asyncio.Lock`

Out:

- Watch mode / daemon
- Multiple patterns per run
- Cross-run lockfile (single-writer assumption)

## CLI

```
python ingest.py [--docs-dir PATH] [--pattern GLOB] [--dry-run] [--concurrency N]
```

Defaults:

- `--docs-dir`: `$SAMPLE_DOCS_DIR` if set, else `./sample_docs`
- `--pattern`: `*.txt`
- `--dry-run`: false
- `--concurrency`: `1`

`Path.glob(pattern)` is used, so callers can pass `**/*.txt` for recursion.

## Diff algorithm

```
disk = {relpath(f): sha256(f) for f in docs_dir.glob(pattern) if f.is_file()}
ledger = load_ledger()  # {relpath: hash}

new       = disk.keys() - ledger.keys()
removed   = ledger.keys() - disk.keys()
changed   = {k for k in disk.keys() & ledger.keys() if disk[k] != ledger[k]}
unchanged = (disk.keys() & ledger.keys()) - changed
```

Ledger key: relative path from `--docs-dir` (POSIX form). Avoids name
collisions across subdirs when callers use recursive globs.

Doc id: `Path(relpath).stem`. (Consistent with `reingest_two.py`.) Subdir
collisions on stem are out of scope; document the constraint.

## Actions per category

| Category  | Action                                           |
|-----------|--------------------------------------------------|
| new       | `ainsert(text, ids=[stem], file_paths=[relpath])` |
| changed   | `adelete_by_doc_id(stem)` then `ainsert(...)`     |
| removed   | `adelete_by_doc_id(stem)`, drop ledger entry      |
| unchanged | skip                                             |

Ledger written under `asyncio.Lock` after each successful task so partial
progress survives crashes.

## Dry-run output

```
plan:
  add:    N files
  update: M files
  delete: K files
  skip:   U files
(no changes made)
```

When N+M+K is small (≤20), list each file under its category.

## Concurrency

- `sem = asyncio.Semaphore(args.concurrency)`
- One coroutine per pending file (new, changed, removed)
- `asyncio.gather(*tasks)` — first exception aborts pending tasks
- Ledger writes serialized via `asyncio.Lock`

`--concurrency 1` preserves current sequential behavior.

## Error handling

- File read failure: abort task, surface exception via `gather`
- `ainsert` / `adelete_by_doc_id` exception: propagate; ledger reflects only
  completed tasks
- Missing docs dir: exit non-zero with clear message
- Empty match: print `nothing to ingest`, exit 0 (not an error)

## Migration

Existing ledger keys are bare filenames. On load, treat any key without `/`
as relative to the dir root — no rewrite needed; new entries use POSIX
relpaths and old entries naturally migrate when files change.

## Verification

- Add file → run → ledger has entry, KG has doc
- Edit file → run → `adelete_by_doc_id` called, fresh entities replace old
- Delete file → run → `adelete_by_doc_id` called, ledger entry gone
- `--dry-run` → no RAG load, no ledger write
- `--concurrency 4` with 4+ pending → tasks overlap (observable via timing)

Existing tests under `tests/` still pass.

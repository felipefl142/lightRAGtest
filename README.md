# lightRAGtest

LightRAG playground: local Ollama + Streamlit chat + needle-in-haystack eval harness.

## Stack

- [LightRAG](https://github.com/HKUDS/LightRAG) — KG + vector RAG
- Ollama — local LLM + embeddings (default: `gemma4` / `nomic-embed-text`)
- Streamlit — chat UI
- `uv` — deps

## Layout

| File | Purpose |
|---|---|
| `rag_core.py` | LightRAG factory + LLM reranker. Single source of truth for config. |
| `ingest.py` | Index every `*.txt` in `sample_docs/`. SHA-256 ledger skips re-indexed files. |
| `app.py` | Streamlit chat UI. Mode switcher, file upload, persistent session history. |
| `fetch_data.py` | Seed corpus: 2 Paul Graham essays + 5 RAG-adjacent Wikipedia pages. |
| `fetch_more.py` | Grow corpus to ~100 random Wikipedia docs + inject synthetic needles. |
| `eval.py` | Needle-in-haystack eval across all LightRAG modes ± LLM reranker. |
| `needles.jsonl` | Gold needle manifest (question / answer / source file). |
| `eval_results.json` | Last eval run output. |
| `tests/` | pytest suites: smoke, unit, integration, functional. |

## Setup

```bash
uv sync
cp .env.example .env       # tweak models / host if needed
ollama pull gemma4
ollama pull nomic-embed-text
```

## Run

```bash
# 1. pull seed corpus
python fetch_data.py

# 2. (optional) grow to ~100 docs and inject 15 needles
python fetch_more.py

# 3. index
python ingest.py

# 4. chat
streamlit run app.py

# 5. eval
python eval.py --rerank
```

## Config

All config via env vars (see `.env.example`). Key knobs in `rag_core.py`:

- `LLM_MODEL`, `EMBED_MODEL`, `EMBED_DIM` — model selection. Embed dim must match model.
- `CHUNK_TOKEN_SIZE=800`, `CHUNK_OVERLAP=80`, `MAX_GLEANING=0` — tuned small to avoid timeouts on local Ollama.
- `LLM_MAX_ASYNC=2`, `EMBED_MAX_ASYNC=4` — low concurrency to avoid GPU thrash.
- `LLM_TIMEOUT=900` — seconds; generous for slow local inference.

## Query modes

LightRAG exposes `naive`, `local`, `global`, `hybrid`, `mix`. UI defaults to `hybrid`. Eval sweeps all five.

## Reranker

`rerank_passages` in `rag_core.py` is an LLM-as-reranker (0-10 scoring via Ollama). Used by `eval.py --rerank`.

## Tests

```bash
uv run pytest              # all
uv run pytest -m smoke     # fast sanity only
```

## Notes

- `rag_storage/` holds the persisted KG + vectors. Delete it to force a full reindex.
- `sample_docs/`, `.env`, `rag_storage/` gitignored.
- `ollama_embed` is unwrapped in `rag_core.py` to bypass the hardcoded 1024-dim check so non-bge-m3 embedders work.

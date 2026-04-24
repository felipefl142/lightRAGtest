"""Shared LightRAG factory. Used by ingest.py and app.py."""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import ollama
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:latest")
RERANK_MODEL = os.getenv("RERANK_MODEL", LLM_MODEL)
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
EMBED_MAX_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "8192"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "32768"))
WORKING_DIR = os.getenv("WORKING_DIR", "./rag_storage")

# Ingest tuning. Smaller chunks + fewer gleaning passes = fewer LLM timeouts
# on slow local ollama. Concurrency kept low to avoid GPU thrash.
CHUNK_TOKEN_SIZE = int(os.getenv("CHUNK_TOKEN_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
MAX_GLEANING = int(os.getenv("MAX_GLEANING", "0"))
LLM_MAX_ASYNC = int(os.getenv("LLM_MAX_ASYNC", "2"))
EMBED_MAX_ASYNC = int(os.getenv("EMBED_MAX_ASYNC", "4"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "900"))


_RERANK_PROMPT = """Score how relevant this passage is to the query on a 0-10 scale.
Output ONLY a single integer 0-10. No words.

Query: {query}

Passage:
{passage}

Score:"""


async def rerank_passages(
    query: str,
    passages: list[str],
    top_k: int = 5,
    model: str | None = None,
) -> list[tuple[int, float, str]]:
    """LLM-as-reranker via Ollama. Returns top_k [(orig_idx, score, passage)].

    Scores each passage 0-10 with small LLM call. Concurrent via asyncio.
    """
    client = ollama.AsyncClient(host=OLLAMA_HOST)
    mdl = model or RERANK_MODEL

    async def score_one(idx: int, passage: str) -> tuple[int, float, str]:
        prompt = _RERANK_PROMPT.format(query=query, passage=passage[:1500])
        try:
            resp = await client.generate(
                model=mdl,
                prompt=prompt,
                options={"num_predict": 4, "temperature": 0.0},
            )
            m = re.search(r"\d+", resp["response"])
            score = float(m.group()) if m else 0.0
        except Exception:
            score = 0.0
        return idx, score, passage

    results = await asyncio.gather(*(score_one(i, p) for i, p in enumerate(passages)))
    results.sort(key=lambda r: r[1], reverse=True)
    return results[:top_k]


async def build_rag() -> LightRAG:
    Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)

    # ollama_embed is wrapped with embedding_dim=1024 for bge-m3.
    # Call .func to bypass that check so our EmbeddingFunc(embedding_dim=EMBED_DIM)
    # is the single source of truth, letting non-1024 models (nomic-embed-text=768)
    # work without a dim mismatch.
    _raw_ollama_embed = getattr(ollama_embed, "func", ollama_embed)

    async def _embed(texts):
        return await _raw_ollama_embed(
            texts, embed_model=EMBED_MODEL, host=OLLAMA_HOST
        )

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={
            "host": OLLAMA_HOST,
            "timeout": LLM_TIMEOUT,
            "options": {
                "num_ctx": LLM_NUM_CTX,
                "temperature": 0.0,
            },
        },
        llm_model_max_async=LLM_MAX_ASYNC,
        embedding_func_max_async=EMBED_MAX_ASYNC,
        chunk_token_size=CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP,
        entity_extract_max_gleaning=MAX_GLEANING,
        default_llm_timeout=LLM_TIMEOUT,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=EMBED_MAX_TOKENS,
            func=_embed,
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

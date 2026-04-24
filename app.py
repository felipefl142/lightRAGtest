"""Streamlit chat UI over LightRAG with persistent KG + session history."""
from __future__ import annotations

import asyncio
from pathlib import Path

import nest_asyncio
import streamlit as st
from lightrag import QueryParam

from rag_core import LLM_MODEL, WORKING_DIR, build_rag

nest_asyncio.apply()

st.set_page_config(page_title="LightRAG Chat", page_icon=":speech_balloon:", layout="wide")

MODES = ["naive", "local", "global", "hybrid", "mix"]


def _loop() -> asyncio.AbstractEventLoop:
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()
    return st.session_state.loop


@st.cache_resource(show_spinner="Initializing LightRAG...")
def get_rag():
    loop = _loop()
    return loop.run_until_complete(build_rag())


def run(coro):
    return _loop().run_until_complete(coro)


def ingest_upload(rag, name: str, text: str) -> None:
    run(rag.ainsert(text, ids=[f"upload::{name}"], file_paths=[name]))


def query(rag, prompt: str, mode: str, history: list[dict]) -> str:
    param = QueryParam(
        mode=mode,
        conversation_history=history,
        history_turns=6,
    )
    return run(rag.aquery(prompt, param=param))


rag = get_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    st.caption(f"LLM: `{LLM_MODEL}`")
    st.caption(f"Store: `{Path(WORKING_DIR).resolve()}`")
    mode = st.selectbox("Query mode", MODES, index=MODES.index("hybrid"))
    st.divider()

    st.subheader("Add document")
    uploaded = st.file_uploader("Plain text / markdown", type=["txt", "md"])
    if uploaded and st.button("Ingest", use_container_width=True):
        text = uploaded.read().decode("utf-8", errors="replace")
        with st.spinner(f"Indexing {uploaded.name}..."):
            ingest_upload(rag, uploaded.name, text)
        st.success(f"Ingested {uploaded.name}")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("LightRAG Chat")
st.caption("KG + vectors persist in `rag_storage/`. Chat history persists per session.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the ingested docs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner(f"Querying ({mode})..."):
            try:
                answer = query(rag, prompt, mode, history)
            except Exception as e:
                answer = f"**Error:** {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

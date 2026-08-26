"""DataPilot AI — Streamlit chat UI (Priority 8)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.model.adapter import resolve_lora_adapter_path
from src.ui.app_support import (
    EXAMPLE_QUESTIONS,
    MODE_LABELS,
    build_chatbot,
    default_mode,
    mode_settings,
    run_turn,
)
from src.utils.config import load_env, load_yaml


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = default_mode()


@st.cache_resource
def _cached_bot(mode: str, top_k: int):
    return build_chatbot(mode, top_k=top_k)


def main() -> None:
    load_env()
    st.set_page_config(page_title="DataPilot AI", page_icon="📊", layout="centered")
    model_cfg = load_yaml("./config/model.yaml")
    selected = model_cfg.get("selected", {})
    _init_state()
    st.title("DataPilot AI")
    st.caption("An AI Copilot for Business Intelligence & Data Engineering")

    with st.sidebar:
        st.subheader("Settings")
        mode_ids = list(MODE_LABELS.keys())
        current = st.session_state.mode if st.session_state.mode in mode_ids else default_mode()
        mode = st.selectbox(
            "System",
            options=mode_ids,
            index=mode_ids.index(current),
            format_func=lambda m: MODE_LABELS[m],
        )
        if mode != st.session_state.mode:
            st.session_state.mode = mode
            st.session_state.messages = []

        top_k = st.slider("Retrieval top-k", min_value=1, max_value=8, value=5)
        adapter = resolve_lora_adapter_path(model_cfg)
        if adapter is not None:
            st.success(f"Adapter found: `{adapter.as_posix()}`")
        else:
            st.warning("No LoRA adapter on disk. System C falls back until the adapter is copied in.")

        st.caption(f"Base model: `{selected.get('model_id', 'n/a')}`")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            try:
                _cached_bot(st.session_state.mode, int(top_k)).memory.clear()
            except Exception:
                pass
            st.rerun()

        with st.expander("About this demo"):
            st.markdown(
                "- Non-agentic V1: no live SQL execution.\n"
                "- Answers may use retrieved official docs; sources are listed when available.\n"
                "- Retrieve-only mode does not load the LLM (useful without a GPU).\n"
                "- Full chat loads Qwen2.5-1.5B on first question (slow on CPU)."
            )

    settings = mode_settings(st.session_state.mode)
    if st.session_state.mode == "finetuned_rag" and settings["adapter_path"] is None:
        st.info("Fine-tuned adapter not found — using RAG only for this session.")
        st.session_state.mode = "rag_only"

    st.markdown("**Example questions**")
    cols = st.columns(min(3, len(EXAMPLE_QUESTIONS)))
    pending = st.session_state.pop("pending_question", None)
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % len(cols)].button(example, key=f"ex_{i}"):
            pending = example

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        title = src.get("title") or "Source"
                        url = src.get("url") or ""
                        score = src.get("score")
                        extra = f" ({float(score):.3f})" if score is not None else ""
                        if url:
                            st.markdown(f"- [{title}]({url}){extra}")
                        else:
                            st.markdown(f"- {title}{extra}")

    question = pending or st.chat_input("Ask a BI or Data Engineering question…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    bot = _cached_bot(st.session_state.mode, int(top_k))
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = run_turn(
                    bot,
                    question,
                    retrieve_only=bool(settings["retrieve_only"]),
                )
            except Exception as exc:  # noqa: BLE001
                result = {
                    "answer": f"Something went wrong: {exc}",
                    "sources": [],
                    "mode": "error",
                    "latency_ms": 0.0,
                }
        answer = result.get("answer") or ""
        st.markdown(answer)
        sources = result.get("sources") or []
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    title = src.get("title") or "Source"
                    url = src.get("url") or ""
                    if url:
                        st.markdown(f"- [{title}]({url})")
                    else:
                        st.markdown(f"- {title}")
        meta = result.get("mode", "")
        latency = result.get("latency_ms") or 0.0
        st.caption(f"mode={meta} · {latency:.0f} ms")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "mode": result.get("mode"),
        }
    )


if __name__ == "__main__":
    main()

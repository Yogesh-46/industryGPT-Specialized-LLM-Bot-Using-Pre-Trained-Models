"""Baseline RAG chatbot: retrieve → prompt → generate → sources."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from src.embeddings.encoder import encoder_from_rag_config
from src.generation.domain import OOD_REFUSAL, is_out_of_domain
from src.generation.memory import ConversationMemory
from src.generation.prompts import (
    build_chat_messages,
    format_sources,
    load_system_prompt,
)
from src.model.llm import HuggingFaceLLM
from src.retrieval.faiss_store import FaissVectorStore
from src.utils.config import load_yaml
from src.utils.paths import resolve_path

logger = logging.getLogger(__name__)

GenerateFn = Callable[[list[dict[str, str]]], str]


class BaselineRAGChatbot:
    """RAG chatbot: Systems B (base+RAG) and C (adapter+RAG)."""

    def __init__(
        self,
        *,
        rag_config: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        adapter_path: str | None = None,
        llm: HuggingFaceLLM | None = None,
        generate_fn: GenerateFn | None = None,
        use_rag: bool = True,
    ) -> None:
        self.rag_config = rag_config or load_yaml("./config/rag.yaml")
        self.model_config = model_config or load_yaml("./config/model.yaml")
        self.use_rag = use_rag

        retrieval = self.rag_config.get("retrieval", {})
        prompting = self.rag_config.get("prompting", {})
        memory_cfg = self.rag_config.get("conversation_memory", {})
        query_cfg = self.rag_config.get("query_processing", {})
        vs_cfg = self.rag_config.get("vector_store", {})

        self.top_k = int(retrieval.get("top_k", 5))
        self.score_threshold = retrieval.get("score_threshold")
        self.max_context_chars = int(prompting.get("max_context_chars", 8000))
        self.max_query_chars = int(query_cfg.get("max_query_chars", 4000))
        self.strip_whitespace = bool(query_cfg.get("strip_whitespace", True))

        self.system_prompt = load_system_prompt(
            prompting.get("system_prompt_path", "./config/prompts/system_prompt.txt")
        )
        self.memory = ConversationMemory(
            max_turns=int(memory_cfg.get("max_turns", 8)),
            max_history_chars=int(memory_cfg.get("max_history_chars", 6000)),
        )

        self.encoder = encoder_from_rag_config(self.rag_config)
        self.store = FaissVectorStore(
            resolve_path(vs_cfg.get("persist_dir", "./knowledge_base/vector_store")),
            index_filename=vs_cfg.get("index_filename", "faiss.index"),
            metadata_filename=vs_cfg.get("metadata_filename", "chunks_metadata.jsonl"),
        )
        self._store_loaded = False

        self.llm = llm
        self.generate_fn = generate_fn
        self.adapter_path = adapter_path

    def _ensure_store(self) -> None:
        if not self._store_loaded:
            self.store.load()
            self._store_loaded = True

    def _ensure_generator(self) -> GenerateFn:
        if self.generate_fn is not None:
            return self.generate_fn
        if self.llm is None:
            self.llm = HuggingFaceLLM.from_configs(
                self.model_config, adapter_path=self.adapter_path
            )
        return self.llm.generate

    def _normalize_query(self, query: str) -> str:
        text = query or ""
        if self.strip_whitespace:
            text = " ".join(text.split())
        if len(text) > self.max_query_chars:
            text = text[: self.max_query_chars]
        return text

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve top-k chunks with scores and metadata."""
        self._ensure_store()
        vector = self.encoder.encode_query(query)
        hits = self.store.search(
            vector,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )
        return [
            {
                "rank": h.rank,
                "score": h.score,
                "metadata": h.metadata,
            }
            for h in hits
        ]

    def ask(self, query: str, *, use_history: bool = True) -> dict[str, Any]:
        """Answer one user query with optional RAG grounding."""
        started = time.perf_counter()
        query = self._normalize_query(query)
        if not query:
            return {
                "answer": "Please enter a question about BI or Data Engineering.",
                "sources": [],
                "hits": [],
                "out_of_domain": False,
                "latency_ms": 0.0,
                "mode": "empty",
            }

        ood, ood_category = is_out_of_domain(query)
        if ood:
            answer = OOD_REFUSAL
            if use_history:
                self.memory.add("user", query)
                self.memory.add("assistant", answer)
            return {
                "answer": answer,
                "sources": [],
                "hits": [],
                "out_of_domain": True,
                "ood_category": ood_category,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "mode": "ood_refusal",
            }

        hits: list[dict[str, Any]] = []
        if self.use_rag:
            try:
                hits = self.retrieve(query)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Retrieval failed")
                return {
                    "answer": f"Retrieval failed: {exc}",
                    "sources": [],
                    "hits": [],
                    "out_of_domain": False,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "mode": "retrieval_error",
                    "error": str(exc),
                }

        history = self.memory.as_chat_messages() if use_history else []
        # History should be prior turns only; current query is added via prompt builder
        messages = build_chat_messages(
            system_prompt=self.system_prompt,
            query=query,
            hits=hits if self.use_rag else [],
            history=history,
            max_context_chars=self.max_context_chars,
        )

        try:
            generate = self._ensure_generator()
            answer = generate(messages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Generation failed")
            return {
                "answer": f"Generation failed: {exc}",
                "sources": format_sources(hits),
                "hits": hits,
                "out_of_domain": False,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "mode": "generation_error",
                "error": str(exc),
            }

        if use_history:
            self.memory.add("user", query)
            self.memory.add("assistant", answer)

        if self.adapter_path and self.use_rag:
            mode = "finetuned_rag"
        elif self.use_rag:
            mode = "rag"
        else:
            mode = "baseline_llm"

        return {
            "answer": answer,
            "sources": format_sources(hits),
            "hits": hits,
            "out_of_domain": False,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "mode": mode,
            "model_id": self.model_config.get("selected", {}).get("model_id"),
            "adapter_path": self.adapter_path,
        }

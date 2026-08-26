"""Reusable evaluation runner for Systems A / B / C."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.evaluation.metrics import score_prediction
from src.evaluation.schema import load_eval_jsonl
from src.generation.rag import BaselineRAGChatbot
from src.model.adapter import resolve_lora_adapter_path
from src.utils.config import load_yaml
from src.utils.paths import ensure_dir, project_root, resolve_path

logger = logging.getLogger(__name__)

SYSTEMS = {
    "baseline_llm": {"use_rag": False, "adapter": False},
    "rag_only": {"use_rag": True, "adapter": False},
    "finetuned_rag": {"use_rag": True, "adapter": True},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(project_root()))
    except ValueError:
        return str(path)


class EvaluationRunner:
    """Run held-out Dataset C against one or more system configurations."""

    def __init__(
        self,
        *,
        eval_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        rag_config: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        adapter_path: str | None = None,
        generate_fn: Callable[[list[dict[str, str]]], str] | None = None,
        compute_semantic_similarity: bool = False,
    ) -> None:
        eval_cfg = load_yaml("./config/evaluation.yaml")
        self.eval_path = resolve_path(
            eval_path
            or eval_cfg.get("datasets", {}).get(
                "evaluation_path", "./data/evaluation/eval_set.jsonl"
            )
        )
        self.output_dir = resolve_path(
            output_dir or "./experiments/results/evaluation"
        )
        self.rag_config = rag_config or load_yaml("./config/rag.yaml")
        self.model_config = model_config or load_yaml("./config/model.yaml")
        self.adapter_path = adapter_path
        self.generate_fn = generate_fn
        self.compute_semantic_similarity = compute_semantic_similarity
        self._encoder = None

    def _get_semantic_similarity(self, answer: str, reference: str) -> float | None:
        if not self.compute_semantic_similarity:
            return None
        try:
            from src.embeddings.encoder import encoder_from_rag_config
            import numpy as np

            if self._encoder is None:
                self._encoder = encoder_from_rag_config(self.rag_config)
                self._encoder.enable_disk_cache = False
            vectors = self._encoder.encode([answer or "", reference or ""])
            if vectors.shape[0] < 2:
                return None
            a, b = vectors[0], vectors[1]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            return float(np.dot(a, b) / denom)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Semantic similarity unavailable: %s", exc)
            return None

    def _build_bot(self, system_id: str) -> BaselineRAGChatbot | None:
        cfg = SYSTEMS[system_id]
        adapter = None
        if cfg["adapter"]:
            resolved = resolve_lora_adapter_path(
                self.model_config, explicit=self.adapter_path
            )
            if resolved is None:
                logger.warning(
                    "Skipping %s: LoRA adapter not found (set paths.adapter_path).",
                    system_id,
                )
                return None
            adapter = str(resolved)
            logger.info("System %s using adapter %s", system_id, adapter)
        return BaselineRAGChatbot(
            rag_config=self.rag_config,
            model_config=self.model_config,
            adapter_path=adapter,
            generate_fn=self.generate_fn,
            use_rag=bool(cfg["use_rag"]),
        )

    def run(
        self,
        *,
        systems: list[str] | None = None,
        limit: int | None = None,
        experiment_id: str = "eval_framework_run",
    ) -> dict[str, Any]:
        systems = systems or ["baseline_llm", "rag_only", "finetuned_rag"]
        items = load_eval_jsonl(self.eval_path)
        if limit is not None:
            items = items[: max(0, limit)]

        ensure_dir(self.output_dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.output_dir / f"{experiment_id}_{stamp}"
        ensure_dir(run_dir)

        all_rows: list[dict[str, Any]] = []
        summaries: dict[str, Any] = {}

        for system_id in systems:
            if system_id not in SYSTEMS:
                raise ValueError(f"Unknown system_id: {system_id}")
            bot = self._build_bot(system_id)
            if bot is None:
                summaries[system_id] = {
                    "status": "skipped",
                    "reason": "missing_adapter_or_unavailable",
                }
                continue

            # Fresh memory per system; disable history during batch eval
            bot.memory.clear()
            system_rows: list[dict[str, Any]] = []

            for item in items:
                result = bot.ask(item["question"], use_history=False)
                contexts = []
                for hit in result.get("hits") or []:
                    meta = hit.get("metadata") or {}
                    if meta.get("content"):
                        contexts.append(str(meta["content"]))

                sem = self._get_semantic_similarity(
                    result.get("answer") or "",
                    item.get("reference_answer") or "",
                )
                metrics = score_prediction(
                    answer=result.get("answer") or "",
                    reference_answer=item.get("reference_answer") or "",
                    expected_answer_points=item.get("expected_answer_points") or [],
                    category=item.get("category") or "",
                    contexts=contexts,
                    latency_ms=result.get("latency_ms"),
                    semantic_similarity=sem,
                )
                row = {
                    "experiment_id": experiment_id,
                    "datetime": _utc_now(),
                    "system_id": system_id,
                    "question_id": item.get("question_id"),
                    "category": item.get("category"),
                    "difficulty": item.get("difficulty"),
                    "question": item.get("question"),
                    "answer": result.get("answer"),
                    "mode": result.get("mode"),
                    "out_of_domain_flag": result.get("out_of_domain"),
                    "sources": result.get("sources") or [],
                    "metrics": metrics,
                    "model_id": result.get("model_id")
                    or self.model_config.get("selected", {}).get("model_id"),
                    "dataset_version": "datapilot_eval_v1",
                }
                system_rows.append(row)
                all_rows.append(row)

            summaries[system_id] = self._summarize(system_rows)
            self._write_system_outputs(run_dir, system_id, system_rows)

        summary = {
            "experiment_id": experiment_id,
            "datetime": _utc_now(),
            "dataset_path": _rel(self.eval_path),
            "dataset_version": "datapilot_eval_v1",
            "n_questions": len(items),
            "systems_requested": systems,
            "system_summaries": summaries,
            "output_dir": _rel(run_dir),
            "notes": [
                "Automatic metrics only; human rubric scores are not fabricated.",
                "auto_score is a heuristic composite for screening, not a final academic grade.",
                "finetuned_rag uses the LoRA adapter from config/model.yaml paths.adapter_path when present.",
            ],
        }
        summary_path = run_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        self._write_combined_csv(run_dir, all_rows)
        logger.info("Evaluation complete: %s", run_dir)
        return summary

    def _summarize(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"status": "empty", "n": 0}

        def avg(key: str) -> float | None:
            vals = []
            for r in rows:
                v = (r.get("metrics") or {}).get(key)
                if v is not None:
                    vals.append(float(v))
            return (sum(vals) / len(vals)) if vals else None

        by_category: dict[str, list[float]] = {}
        for r in rows:
            cat = str(r.get("category"))
            score = (r.get("metrics") or {}).get("auto_score")
            if score is None:
                continue
            by_category.setdefault(cat, []).append(float(score))

        return {
            "status": "completed",
            "n": len(rows),
            "mean_point_coverage": avg("point_coverage"),
            "mean_token_f1": avg("token_f1"),
            "mean_rouge_l": avg("rouge_l"),
            "mean_groundedness_proxy": avg("groundedness_proxy"),
            "mean_retrieval_relevance_proxy": avg("retrieval_relevance_proxy"),
            "mean_ood_refusal_score": avg("ood_refusal_score"),
            "mean_semantic_similarity": avg("semantic_similarity"),
            "mean_latency_ms": avg("latency_ms"),
            "mean_auto_score": avg("auto_score"),
            "mean_auto_score_by_category": {
                k: (sum(v) / len(v)) for k, v in by_category.items()
            },
        }

    def _write_system_outputs(
        self, run_dir: Path, system_id: str, rows: list[dict[str, Any]]
    ) -> None:
        jsonl_path = run_dir / f"{system_id}_predictions.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_combined_csv(self, run_dir: Path, rows: list[dict[str, Any]]) -> None:
        csv_path = run_dir / "all_predictions.csv"
        fieldnames = [
            "experiment_id",
            "datetime",
            "system_id",
            "question_id",
            "category",
            "difficulty",
            "mode",
            "latency_ms",
            "point_coverage",
            "token_f1",
            "rouge_l",
            "groundedness_proxy",
            "retrieval_relevance_proxy",
            "ood_refusal_score",
            "semantic_similarity",
            "auto_score",
            "model_id",
            "dataset_version",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                m = row.get("metrics") or {}
                writer.writerow(
                    {
                        "experiment_id": row.get("experiment_id"),
                        "datetime": row.get("datetime"),
                        "system_id": row.get("system_id"),
                        "question_id": row.get("question_id"),
                        "category": row.get("category"),
                        "difficulty": row.get("difficulty"),
                        "mode": row.get("mode"),
                        "latency_ms": m.get("latency_ms"),
                        "point_coverage": m.get("point_coverage"),
                        "token_f1": m.get("token_f1"),
                        "rouge_l": m.get("rouge_l"),
                        "groundedness_proxy": m.get("groundedness_proxy"),
                        "retrieval_relevance_proxy": m.get("retrieval_relevance_proxy"),
                        "ood_refusal_score": m.get("ood_refusal_score"),
                        "semantic_similarity": m.get("semantic_similarity"),
                        "auto_score": m.get("auto_score"),
                        "model_id": row.get("model_id"),
                        "dataset_version": row.get("dataset_version"),
                    }
                )

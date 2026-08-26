"""Retrieval-only experiments on Dataset C (no LLM generation)."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from collections.abc import Callable

from src.evaluation.metrics import retrieval_relevance_proxy
from src.evaluation.schema import load_eval_jsonl
from src.generation.rag import BaselineRAGChatbot
from src.utils.paths import ensure_dir, project_root, resolve_path


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _mean(vals: list[float]) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    ordered = sorted(vals)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def run_retrieval_study(
    *,
    eval_path: str | Path = "./data/evaluation/eval_set.jsonl",
    output_dir: str | Path = "./experiments/results/evaluation",
    top_k_values: tuple[int, ...] = (3, 5, 8),
    experiment_id: str = "exp_05_topk_retrieval",
    limit: int | None = None,
    retrieve_fn: Callable[[str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Score FAISS retrieval against expected answer points (Dataset C).

    Out-of-domain items are reported separately and excluded from in-domain means.
    Pass ``retrieve_fn`` in tests to avoid loading FAISS.
    """
    items = load_eval_jsonl(resolve_path(eval_path))
    if limit is not None:
        items = items[: max(0, limit)]
    if retrieve_fn is None:
        bot = BaselineRAGChatbot(use_rag=True, generate_fn=lambda _m: "")
        max_k = max(int(k) for k in top_k_values)
        bot.top_k = max_k
        retrieve_fn = bot.retrieve
    started = time.perf_counter()
    max_k = max(int(k) for k in top_k_values)

    # One retrieve per question at max_k; slice for smaller k.
    cached: list[tuple[dict[str, Any], list[dict[str, Any]], float]] = []
    for item in items:
        q_start = time.perf_counter()
        hits = retrieve_fn(item["question"])[:max_k]
        latency_ms = (time.perf_counter() - q_start) * 1000
        cached.append((item, hits, latency_ms))

    per_k: dict[str, Any] = {}
    for k in top_k_values:
        k_int = int(k)
        rows: list[dict[str, Any]] = []
        in_domain: list[float] = []
        ood_scores: list[float] = []
        latencies: list[float] = []
        by_category: dict[str, list[float]] = defaultdict(list)
        by_difficulty: dict[str, list[float]] = defaultdict(list)

        for item, hits, latency_ms in cached:
            sliced = hits[:k_int]
            contexts = [
                str((h.get("metadata") or {}).get("content") or "")
                for h in sliced
                if (h.get("metadata") or {}).get("content")
            ]
            score = retrieval_relevance_proxy(
                item.get("expected_answer_points") or [],
                contexts,
            )
            category = str(item.get("category") or "")
            row = {
                "question_id": item.get("question_id"),
                "category": category,
                "difficulty": item.get("difficulty"),
                "top_k": k_int,
                "n_hits": len(sliced),
                "retrieval_relevance_proxy": score,
                "latency_ms": round(latency_ms, 2),
            }
            rows.append(row)
            latencies.append(float(latency_ms))
            if score is None:
                continue
            if category == "Out-of-domain":
                ood_scores.append(float(score))
            else:
                in_domain.append(float(score))
                by_category[category].append(float(score))
                by_difficulty[str(item.get("difficulty") or "")].append(float(score))

        per_k[str(k_int)] = {
            "n": len(rows),
            "n_in_domain": len(in_domain),
            "n_ood": len(ood_scores),
            "n_zero_coverage_in_domain": sum(1 for v in in_domain if v == 0.0),
            "mean_retrieval_relevance_in_domain": _mean(in_domain),
            "mean_retrieval_relevance_ood": _mean(ood_scores),
            "mean_latency_ms": _mean(latencies),
            "median_latency_ms": _median(latencies),
            "mean_by_category": {c: _mean(v) for c, v in sorted(by_category.items())},
            "mean_by_difficulty": {d: _mean(v) for d, v in sorted(by_difficulty.items())},
            "rows": rows,
        }

    elapsed = time.perf_counter() - started
    run_dir = ensure_dir(resolve_path(output_dir) / f"{experiment_id}_{_utc_stamp()}")
    summary = {
        "experiment_id": experiment_id,
        "status": "completed",
        "kind": "retrieval_only",
        "uses_llm": False,
        "dataset_path": "data/evaluation/eval_set.jsonl",
        "dataset_version": "datapilot_eval_v1",
        "n_questions": len(items),
        "top_k_values": list(top_k_values),
        "wall_clock_seconds": round(elapsed, 2),
        "per_top_k": {
            k: {kk: vv for kk, vv in blob.items() if kk != "rows"}
            for k, blob in per_k.items()
        },
        "academic_note": (
            "Retrieval relevance proxy is lexical coverage of expected answer "
            "points in retrieved chunks. It is not a human grade and does not "
            "measure generation quality. Out-of-domain items are excluded from "
            "in-domain means."
        ),
        "completed_at_utc": _utc_stamp(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    # Flatten rows for all k
    all_rows: list[dict[str, Any]] = []
    for blob in per_k.values():
        all_rows.extend(blob["rows"])
    with (run_dir / "retrieval_rows.jsonl").open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        summary["output_dir"] = str(run_dir.relative_to(project_root())).replace("\\", "/")
    except ValueError:
        summary["output_dir"] = str(run_dir)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary

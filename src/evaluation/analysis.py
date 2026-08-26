"""Analyse executed experiment artefacts (no fabricated scores)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.schema import load_eval_jsonl
from src.utils.paths import ensure_dir, resolve_path

OOD_CATEGORY = "Out-of-domain"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def enrich_retrieval_summary(
    summary: dict[str, Any] | None,
    retrieval_dir: str | Path | None,
) -> dict[str, Any] | None:
    """Fill latency / OOD / zero-coverage from retrieval_rows.jsonl if missing."""
    if not summary or not retrieval_dir:
        return summary
    rows = load_jsonl(Path(retrieval_dir) / "retrieval_rows.jsonl")
    if not rows:
        return summary
    by_k: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_k[str(row.get("top_k"))].append(row)
    per_k = dict(summary.get("per_top_k") or {})
    for k, k_rows in by_k.items():
        blob = dict(per_k.get(k) or {})
        in_domain = [
            float(r["retrieval_relevance_proxy"])
            for r in k_rows
            if r.get("category") != OOD_CATEGORY
            and r.get("retrieval_relevance_proxy") is not None
        ]
        ood = [
            float(r["retrieval_relevance_proxy"])
            for r in k_rows
            if r.get("category") == OOD_CATEGORY
            and r.get("retrieval_relevance_proxy") is not None
        ]
        latencies = [float(r["latency_ms"]) for r in k_rows if r.get("latency_ms") is not None]
        blob.setdefault("n", len(k_rows))
        blob.setdefault("n_in_domain", len(in_domain))
        blob["n_ood"] = len(ood)
        blob["n_zero_coverage_in_domain"] = sum(1 for v in in_domain if v == 0.0)
        blob.setdefault("mean_retrieval_relevance_in_domain", _mean(in_domain))
        blob["mean_retrieval_relevance_ood"] = _mean(ood)
        blob["mean_latency_ms"] = _mean(latencies)
        blob["median_latency_ms"] = _median(latencies)
        per_k[k] = blob
    summary = dict(summary)
    summary["per_top_k"] = per_k
    return summary


def write_retrieval_failure_cases(
    *,
    retrieval_dir: str | Path,
    eval_path: str | Path = "./data/evaluation/eval_set.jsonl",
    top_k: int = 5,
    output_path: str | Path | None = None,
    max_score: float = 0.0,
) -> Path | None:
    """List in-domain Dataset C items with weak retrieval coverage (executed run)."""
    retrieval_dir_p = Path(retrieval_dir)
    rows = [
        r
        for r in load_jsonl(retrieval_dir_p / "retrieval_rows.jsonl")
        if int(r.get("top_k") or 0) == int(top_k)
        and r.get("category") != OOD_CATEGORY
        and r.get("retrieval_relevance_proxy") is not None
        and float(r["retrieval_relevance_proxy"]) <= max_score
    ]
    questions = {
        item["question_id"]: item
        for item in load_eval_jsonl(resolve_path(eval_path))
    }
    out = Path(output_path) if output_path else retrieval_dir_p / "failure_cases.md"
    lines = [
        f"# Retrieval failure cases (top-k={top_k})",
        "",
        "In-domain Dataset C questions whose retrieved chunks covered **none** ",
        "of the expected answer points (lexical proxy). This is not a human grade.",
        "",
        f"Count: **{len(rows)}**",
        "",
    ]
    if not rows:
        lines.append("No in-domain items had zero coverage at this top-k.")
    else:
        lines += [
            "| question_id | category | difficulty | score | question |",
            "|-------------|----------|------------|------:|----------|",
        ]
        for row in sorted(rows, key=lambda r: str(r.get("question_id"))):
            qid = str(row.get("question_id"))
            item = questions.get(qid) or {}
            question = str(item.get("question") or "").replace("|", "/")
            lines.append(
                f"| `{qid}` | {row.get('category')} | {row.get('difficulty')} | "
                f"{float(row['retrieval_relevance_proxy']):.3f} | {question} |"
            )
    ensure_dir(out.parent)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out

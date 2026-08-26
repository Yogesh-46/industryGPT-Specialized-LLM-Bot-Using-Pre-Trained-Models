"""Automatic evaluation metrics for DataPilot AI (no fabricated human scores)."""

from __future__ import annotations

import re
from typing import Any, Sequence


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def point_coverage(answer: str, expected_points: Sequence[str]) -> float:
    """Fraction of expected answer points evidenced in the answer (lexical).

    A point counts as covered if a substantial token subset appears in the answer.
    This is a heuristic proxy for completeness/correctness cues — not a human grade.
    """
    if not expected_points:
        return 0.0
    ans = _normalize(answer)
    if not ans:
        return 0.0
    hits = 0
    for point in expected_points:
        tokens = [t for t in _normalize(point).split() if len(t) > 2]
        if not tokens:
            continue
        # Require at least half of informative tokens
        matched = sum(1 for t in tokens if t in ans)
        if matched / max(len(tokens), 1) >= 0.5:
            hits += 1
    return hits / len(expected_points)


def token_f1(prediction: str, reference: str) -> float:
    """Simple token-level F1 between prediction and reference."""
    pred = _normalize(prediction).split()
    ref = _normalize(reference).split()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for t in pred:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t, c in pred_counts.items():
        overlap += min(c, ref_counts.get(t, 0))
    precision = overlap / max(len(pred), 1)
    recall = overlap / max(len(ref), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure; uses rouge-score if installed, else LCS fallback."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(reference or "", prediction or "")["rougeL"].fmeasure)
    except Exception:  # noqa: BLE001
        return _lcs_f1(prediction, reference)


def _lcs_f1(prediction: str, reference: str) -> float:
    pred = _normalize(prediction).split()
    ref = _normalize(reference).split()
    if not pred or not ref:
        return 0.0
    m, n = len(pred), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / m
    recall = lcs / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def groundedness_proxy(answer: str, contexts: Sequence[str]) -> float | None:
    """Lexical overlap of answer tokens with retrieved context.

    Returns None when no context is available (System A / failed retrieval).
    """
    if not contexts:
        return None
    ctx = _normalize(" ".join(contexts))
    ans_tokens = [t for t in _normalize(answer).split() if len(t) > 2]
    if not ans_tokens:
        return 0.0
    overlap = sum(1 for t in ans_tokens if t in ctx)
    return overlap / len(ans_tokens)


def retrieval_relevance_proxy(
    expected_points: Sequence[str],
    contexts: Sequence[str],
) -> float | None:
    """Proxy for whether retrieved context covers expected answer points."""
    if not contexts:
        return None
    return point_coverage(" ".join(contexts), expected_points)


def ood_refusal_score(answer: str, *, category: str) -> float | None:
    """For out-of-domain items: 1 if answer refuses/scopes out, else 0."""
    if category != "Out-of-domain":
        return None
    ans = _normalize(answer)
    cues = [
        "can't reliably",
        "cannot reliably",
        "out of domain",
        "out-of-domain",
        "specialized in business intelligence",
        "data engineering assistant",
        "can't help with",
        "cannot help with",
        "outside my scope",
        "not able to",
        "i'm datapilot",
    ]
    return 1.0 if any(c in ans for c in cues) else 0.0


def score_prediction(
    *,
    answer: str,
    reference_answer: str,
    expected_answer_points: Sequence[str],
    category: str,
    contexts: Sequence[str] | None = None,
    latency_ms: float | None = None,
    semantic_similarity: float | None = None,
) -> dict[str, Any]:
    """Compute automatic metrics for one prediction."""
    contexts = contexts or []
    metrics = {
        "point_coverage": point_coverage(answer, expected_answer_points),
        "token_f1": token_f1(answer, reference_answer),
        "rouge_l": rouge_l(answer, reference_answer),
        "groundedness_proxy": groundedness_proxy(answer, contexts),
        "retrieval_relevance_proxy": retrieval_relevance_proxy(
            expected_answer_points, contexts
        ),
        "ood_refusal_score": ood_refusal_score(answer, category=category),
        "semantic_similarity": semantic_similarity,
        "latency_ms": latency_ms,
    }
    # Lightweight composite for ranking/smoke only (not a human rubric)
    components = [
        metrics["point_coverage"],
        metrics["token_f1"],
        metrics["rouge_l"],
    ]
    if metrics["ood_refusal_score"] is not None:
        components = [metrics["ood_refusal_score"]]
    metrics["auto_score"] = sum(components) / max(len(components), 1)
    return metrics

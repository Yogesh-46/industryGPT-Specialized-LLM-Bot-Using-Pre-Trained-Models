"""Unit tests for evaluation metrics and runner (mocked LLM)."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.metrics import (
    ood_refusal_score,
    point_coverage,
    rouge_l,
    score_prediction,
    token_f1,
)
from src.evaluation.runner import EvaluationRunner
from src.generation.domain import OOD_REFUSAL


def test_point_coverage_and_token_f1() -> None:
    answer = "LEFT JOIN can duplicate rows when one left row matches many right rows."
    points = [
        "One left row can match many right rows",
        "Duplicates come from one-to-many matches, not LEFT itself",
    ]
    assert point_coverage(answer, points) > 0.4
    assert token_f1(answer, answer) == 1.0
    assert rouge_l(answer, answer) > 0.9


def test_ood_refusal_score() -> None:
    assert ood_refusal_score(OOD_REFUSAL, category="Out-of-domain") == 1.0
    assert ood_refusal_score("Take this medication daily.", category="Out-of-domain") == 0.0
    assert ood_refusal_score("x", category="SQL") is None


def test_score_prediction_bundle() -> None:
    metrics = score_prediction(
        answer="DISTKEY distributes rows across nodes for joins.",
        reference_answer="DISTKEY helps determine how Redshift distributes rows across nodes.",
        expected_answer_points=["Controls data distribution across nodes", "Impacts join co-location"],
        category="Data Warehousing",
        contexts=["DISTKEY determines distribution of rows across compute nodes."],
        latency_ms=12.5,
    )
    assert 0.0 <= metrics["point_coverage"] <= 1.0
    assert metrics["groundedness_proxy"] is not None
    assert metrics["latency_ms"] == 12.5


def test_evaluation_runner_mock(tmp_path: Path) -> None:
    def mock_generate(messages):  # noqa: ANN001
        user = messages[-1]["content"]
        if "diabetes" in user.lower() or "medical" in user.lower():
            # ask() OOD path usually intercepts before generate; keep safe fallback
            return OOD_REFUSAL
        return (
            "DISTKEY distributes rows across nodes. "
            "SORTKEY defines on-disk sort order for filtered scans."
        )

    runner = EvaluationRunner(
        eval_path=Path("data/evaluation/eval_set.jsonl"),
        output_dir=tmp_path / "eval_out",
        generate_fn=mock_generate,
        compute_semantic_similarity=False,
    )
    # Include an OOD item by limiting to a slice that we control via category filter workaround:
    # run first 3 SQL-ish + force systems rag_only only
    summary = runner.run(
        systems=["rag_only", "finetuned_rag"],
        limit=3,
        experiment_id="unit_mock_eval",
    )
    assert summary["n_questions"] == 3
    assert summary["system_summaries"]["rag_only"]["status"] == "completed"
    assert summary["system_summaries"]["rag_only"]["n"] == 3
    run_dirs = list((tmp_path / "eval_out").glob("unit_mock_eval_*"))
    assert run_dirs, "expected output directory"
    assert (run_dirs[0] / "summary.json").exists()
    assert (run_dirs[0] / "rag_only_predictions.jsonl").exists()
    assert (run_dirs[0] / "all_predictions.csv").exists()
    ft = summary["system_summaries"]["finetuned_rag"]
    assert ft["status"] in {"completed", "skipped"}
    if ft["status"] == "completed":
        assert ft["n"] == 3
        assert (run_dirs[0] / "finetuned_rag_predictions.jsonl").exists()

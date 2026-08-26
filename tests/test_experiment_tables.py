"""Tests for experiment tables, retrieval study, and result artefacts (no GPU)."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.analysis import enrich_retrieval_summary, write_retrieval_failure_cases
from src.evaluation.human_eval import write_human_eval_template
from src.evaluation.retrieval_study import run_retrieval_study
from src.evaluation.tables import PLACEHOLDER, render_markdown


def test_results_markdown_uses_training_and_placeholders() -> None:
    md = render_markdown(
        {
            "training": {
                "run_name": "colab_t4_qlora_v1",
                "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "method": "LoRA-fp16",
                "gpu_name": "Tesla T4",
                "train_examples": 600,
                "validation_examples": 67,
                "num_epochs_configured": 3.0,
                "wall_clock_seconds": 977.28,
                "max_memory_allocated_gb": 4.558,
                "train_loss": 0.825,
                "metrics_from_trainer": {
                    "best_eval_loss": 0.595,
                    "eval_loss_by_epoch": [1.01, 0.67, 0.595],
                },
            },
            "retrieval": None,
            "exp_01": None,
            "exp_02": None,
        }
    )
    assert "LoRA-fp16" in md
    assert "Tesla T4" in md
    assert PLACEHOLDER in md
    assert "exp_01" in md


def test_results_markdown_retrieval_difficulty() -> None:
    md = render_markdown(
        {
            "training": {},
            "retrieval": {
                "n_questions": 4,
                "per_top_k": {
                    "5": {
                        "n_in_domain": 3,
                        "mean_retrieval_relevance_in_domain": 0.5,
                        "n_zero_coverage_in_domain": 1,
                        "median_latency_ms": 12.0,
                        "mean_by_category": {"SQL": 1.0},
                        "mean_by_difficulty": {"easy": 1.0, "hard": 0.0},
                        "n_ood": 1,
                        "mean_retrieval_relevance_ood": 0.0,
                    }
                },
            },
            "exp_01": None,
            "exp_02": None,
        }
    )
    assert "Table 2c" in md
    assert "easy" in md
    assert "Out-of-domain" in md


def test_retrieval_study_with_injected_retriever(tmp_path: Path) -> None:
    def retrieve(_question: str) -> list[dict]:
        return [
            {"metadata": {"content": "SELECT retrieves rows from tables. WHERE filters rows."}},
            {"metadata": {"content": "JOIN combines rows from two tables."}},
        ]

    summary = run_retrieval_study(
        eval_path=Path("data/evaluation/eval_set.jsonl"),
        output_dir=tmp_path / "eval",
        top_k_values=(2,),
        experiment_id="unit_retrieval",
        limit=2,
        retrieve_fn=retrieve,
    )
    assert summary["status"] == "completed"
    assert summary["uses_llm"] is False
    assert summary["n_questions"] == 2
    blob = summary["per_top_k"]["2"]
    assert blob["n"] == 2
    assert blob["mean_retrieval_relevance_in_domain"] is not None
    run_dirs = list((tmp_path / "eval").glob("unit_retrieval_*"))
    assert (run_dirs[0] / "retrieval_rows.jsonl").exists()


def test_enrich_and_failure_cases(tmp_path: Path) -> None:
    run = tmp_path / "exp_05_topk_retrieval_unit"
    run.mkdir()
    rows = [
        '{"question_id":"eval_sql_001","category":"SQL","difficulty":"easy","top_k":5,'
        '"n_hits":5,"retrieval_relevance_proxy":0.0,"latency_ms":10.0}\n',
        '{"question_id":"eval_ood_001","category":"Out-of-domain","difficulty":"easy",'
        '"top_k":5,"n_hits":5,"retrieval_relevance_proxy":0.2,"latency_ms":11.0}\n',
    ]
    (run / "retrieval_rows.jsonl").write_text("".join(rows), encoding="utf-8")
    enriched = enrich_retrieval_summary({"per_top_k": {}}, run)
    assert enriched is not None
    blob = enriched["per_top_k"]["5"]
    assert blob["n_in_domain"] == 1
    assert blob["n_ood"] == 1
    assert blob["n_zero_coverage_in_domain"] == 1
    out = write_retrieval_failure_cases(retrieval_dir=run, output_path=tmp_path / "fail.md")
    text = out.read_text(encoding="utf-8")
    assert "eval_sql_001" in text
    assert "eval_ood_001" not in text


def test_human_eval_template_is_blank(tmp_path: Path) -> None:
    import csv

    path = write_human_eval_template(
        output_path=tmp_path / "rubric_blank.csv",
        systems=("rag_only",),
    )
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert all(r["system_id"] == "rag_only" for r in rows)
    assert all(r["correctness_1to5"] == "" for r in rows)
    assert all(r["groundedness_1to5"] == "" for r in rows)

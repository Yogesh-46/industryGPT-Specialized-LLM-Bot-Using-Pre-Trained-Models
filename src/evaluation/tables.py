"""Build thesis-ready tables from real experiment artefacts only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.analysis import enrich_retrieval_summary, write_retrieval_failure_cases
from src.evaluation.figures import write_experiment_figures
from src.evaluation.human_eval import write_human_eval_template
from src.utils.paths import ensure_dir, resolve_path

PLACEHOLDER = "[To be populated after experiment]"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return PLACEHOLDER
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _is_failed_generation_run(run_dir: Path | None) -> bool:
    """Reject runs whose answers are generation_error strings (not model output)."""
    if run_dir is None or not run_dir.is_dir():
        return False
    for jsonl in run_dir.glob("*_predictions.jsonl"):
        try:
            line = jsonl.read_text(encoding="utf-8").splitlines()[0]
            row = json.loads(line)
        except (OSError, json.JSONDecodeError, IndexError):
            continue
        mode = str(row.get("mode") or "")
        answer = str(row.get("answer") or "")
        if mode == "generation_error" or answer.startswith("Generation failed:"):
            return True
    return False


def _is_mock_run(path: Path, summary: dict[str, Any] | None) -> bool:
    name = path.name.lower()
    if "mock" in name:
        return True
    if summary and "mock" in str(summary.get("experiment_id", "")).lower():
        return True
    return False


def latest_matching(root: Path, prefix: str) -> Path | None:
    if not root.is_dir():
        return None
    runs = sorted(
        [
            p
            for p in root.iterdir()
            if p.is_dir() and p.name.startswith(prefix) and "mock" not in p.name.lower()
        ],
        key=lambda p: p.name,
    )
    return runs[-1] if runs else None


def latest_valid_eval(root: Path, prefix: str) -> tuple[Path | None, dict[str, Any] | None]:
    if not root.is_dir():
        return None, None
    runs = sorted(
        [
            p
            for p in root.iterdir()
            if p.is_dir() and p.name.startswith(prefix) and "mock" not in p.name.lower()
        ],
        key=lambda p: p.name,
        reverse=True,
    )
    for run_dir in runs:
        summary = _load_json(run_dir / "summary.json")
        if _is_mock_run(run_dir, summary) or _is_failed_generation_run(run_dir):
            continue
        return run_dir, summary
    return None, None


def collect_results(
    *,
    eval_root: str | Path = "./experiments/results/evaluation",
    training_summary: str | Path = (
        "./experiments/results/adapters/colab_t4_qlora_v1/training_summary.json"
    ),
) -> dict[str, Any]:
    eval_root_p = resolve_path(eval_root)
    train = _load_json(resolve_path(training_summary))
    retrieval_dir = latest_matching(eval_root_p, "exp_05_topk_retrieval")
    retrieval = _load_json(retrieval_dir / "summary.json") if retrieval_dir else None
    retrieval = enrich_retrieval_summary(retrieval, retrieval_dir)
    exp01_dir, exp01 = latest_valid_eval(eval_root_p, "exp_01_baseline_vs_rag")
    exp02_dir, exp02 = latest_valid_eval(eval_root_p, "exp_02_rag_vs_finetuned_rag")

    return {
        "training": train,
        "retrieval": retrieval,
        "retrieval_dir": str(retrieval_dir) if retrieval_dir else None,
        "exp_01": exp01,
        "exp_01_dir": str(exp01_dir) if exp01_dir else None,
        "exp_02": exp02,
        "exp_02_dir": str(exp02_dir) if exp02_dir else None,
    }


def render_markdown(bundle: dict[str, Any], figure_paths: list[str] | None = None) -> str:
    train = bundle.get("training") or {}
    retrieval = bundle.get("retrieval") or {}
    exp01 = bundle.get("exp_01")
    exp02 = bundle.get("exp_02")
    figure_paths = figure_paths or []

    gpu_note = (
        "Tables 3-4 are from executed Colab T4 runs (fp16 inference; not mock LLM)."
        if exp01 or exp02
        else (
            "LLM system comparison cells stay "
            f"`{PLACEHOLDER}` until exp_01 / exp_02 GPU runs finish."
        )
    )
    lines = [
        "# DataPilot AI — experiment results",
        "",
        "Numbers below come only from executed runs. " + gpu_note,
        "",
        "## Table 1 - Fine-tuning (executed, Colab T4)",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Run | `{train.get('run_name', PLACEHOLDER)}` |",
        f"| Model | `{train.get('model_id', PLACEHOLDER)}` |",
        f"| Method | {train.get('method', PLACEHOLDER)} |",
        f"| GPU | {train.get('gpu_name', PLACEHOLDER)} |",
        f"| Train / val examples | {train.get('train_examples', PLACEHOLDER)} / {train.get('validation_examples', PLACEHOLDER)} |",
        f"| Epochs configured | {train.get('num_epochs_configured', PLACEHOLDER)} |",
        f"| Wall clock (s) | {_fmt(train.get('wall_clock_seconds'), 1)} |",
        f"| Peak GPU memory (GiB) | {_fmt(train.get('max_memory_allocated_gb'), 3)} |",
        f"| Train loss (Trainer) | {_fmt(train.get('train_loss'), 4)} |",
        f"| Best eval loss | {_fmt((train.get('metrics_from_trainer') or {}).get('best_eval_loss'), 4)} |",
        f"| Eval loss by epoch | {_fmt((train.get('metrics_from_trainer') or {}).get('eval_loss_by_epoch'))} |",
        "",
        "## Table 2 - Retrieval on Dataset C (executed, no LLM)",
        "",
    ]
    if not retrieval:
        lines += [
            f"Status: {PLACEHOLDER}",
            "",
        ]
    else:
        lines += [
            f"Dataset: {retrieval.get('n_questions')} held-out questions. "
            "Metric: mean retrieval-relevance proxy on **in-domain** items.",
            "",
            "| top-k | In-domain n | Mean retrieval relevance | Zero-coverage n | Median retrieve latency (ms) |",
            "|------:|------------:|-------------------------:|----------------:|-----------------------------:|",
        ]
        per_k = retrieval.get("per_top_k") or {}
        for k in sorted(per_k.keys(), key=lambda x: int(x)):
            blob = per_k[k]
            lines.append(
                f"| {k} | {blob.get('n_in_domain')} | "
                f"{_fmt(blob.get('mean_retrieval_relevance_in_domain'))} | "
                f"{blob.get('n_zero_coverage_in_domain', PLACEHOLDER)} | "
                f"{_fmt(blob.get('median_latency_ms'), 1)} |"
            )
        k5 = per_k.get("5") or {}
        if k5.get("mean_by_category"):
            lines += [
                "",
                "### Table 2b - top-k=5 by category (in-domain)",
                "",
                "| Category | Mean retrieval relevance |",
                "|----------|-------------------------:|",
            ]
            for cat, val in (k5.get("mean_by_category") or {}).items():
                lines.append(f"| {cat} | {_fmt(val)} |")
        if k5.get("mean_by_difficulty"):
            lines += [
                "",
                "### Table 2c - top-k=5 by difficulty (in-domain)",
                "",
                "| Difficulty | Mean retrieval relevance |",
                "|------------|-------------------------:|",
            ]
            order = ["easy", "medium", "hard"]
            diffs = k5.get("mean_by_difficulty") or {}
            for diff in order:
                if diff in diffs:
                    lines.append(f"| {diff} | {_fmt(diffs[diff])} |")
            for diff, val in diffs.items():
                if diff not in order:
                    lines.append(f"| {diff} | {_fmt(val)} |")
        if k5.get("mean_retrieval_relevance_ood") is not None:
            lines += [
                "",
                "### Table 2d - Out-of-domain retrieval (not in in-domain means)",
                "",
                f"OOD n={k5.get('n_ood')}; mean retrieval-relevance proxy "
                f"{_fmt(k5.get('mean_retrieval_relevance_ood'))}. "
                "OOD items should still be refused by the chatbot even if chunks are retrieved.",
                "",
            ]
        lines.append("")

    def _system_row(summary: dict[str, Any] | None, system_id: str) -> str:
        if not summary:
            return (
                f"| `{system_id}` | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | "
                f"{PLACEHOLDER} | {PLACEHOLDER} |"
            )
        blob = (summary.get("system_summaries") or {}).get(system_id) or {}
        if blob.get("status") != "completed":
            return (
                f"| `{system_id}` | {blob.get('status', PLACEHOLDER)} | {PLACEHOLDER} | "
                f"{PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} |"
            )
        return (
            f"| `{system_id}` | {blob.get('n')} | "
            f"{_fmt(blob.get('mean_point_coverage'))} | "
            f"{_fmt(blob.get('mean_token_f1'))} | "
            f"{_fmt(blob.get('mean_rouge_l'))} | "
            f"{_fmt(blob.get('mean_latency_ms'), 1)} |"
        )

    lines += [
        "## Table 3 - exp_01: baseline LLM vs RAG (executed, Colab T4)",
        "",
        "| System | n | Mean point coverage | Mean token F1 | Mean ROUGE-L | Mean latency (ms) |",
        "|--------|--:|--------------------:|--------------:|-------------:|------------------:|",
        _system_row(exp01, "baseline_llm"),
        _system_row(exp01, "rag_only"),
        "",
        "## Table 4 - exp_02: RAG vs fine-tuned + RAG (executed, Colab T4)",
        "",
        "| System | n | Mean point coverage | Mean token F1 | Mean ROUGE-L | Mean latency (ms) |",
        "|--------|--:|--------------------:|--------------:|-------------:|------------------:|",
        _system_row(exp02, "rag_only"),
        _system_row(exp02, "finetuned_rag"),
        "",
        "## Figures",
        "",
    ]
    if figure_paths:
        for fp in figure_paths:
            name = Path(fp).name
            lines.append(f"- `{name}` -- `experiments/results/tables/figures/{name}`")
    else:
        lines.append(f"- {PLACEHOLDER} (run `python scripts/build_results_tables.py`)")
    lines += [
        "",
        "## Integrity",
        "",
        "- Mock LLM smoke tests are **not** copied into Tables 3-4.",
        "- Human 1-5 ratings are not fabricated. Blank rubric: "
        "`experiments/results/human_eval/rubric_blank.csv`.",
        "- Inference for Tables 3-4 used fp16 on Tesla T4 (`load_in_4bit: false`) after bitsandbytes was unavailable.",
        "- Keyword OOD filter refused 5/100 items; Dataset C has 10 Out-of-domain questions.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_results_tables(
    output_dir: str | Path = "./experiments/results/tables",
) -> Path:
    bundle = collect_results()
    out = ensure_dir(resolve_path(output_dir))
    figures = write_experiment_figures(bundle=bundle, output_dir=out / "figures")
    if bundle.get("retrieval_dir"):
        write_retrieval_failure_cases(retrieval_dir=bundle["retrieval_dir"])
    write_human_eval_template()
    md_path = out / "results.md"
    md_path.write_text(render_markdown(bundle, figures), encoding="utf-8")
    (out / "results_bundle.json").write_text(
        json.dumps(
            {
                "has_training": bundle.get("training") is not None,
                "has_retrieval": bundle.get("retrieval") is not None,
                "has_exp_01": bundle.get("exp_01") is not None,
                "has_exp_02": bundle.get("exp_02") is not None,
                "retrieval_dir": bundle.get("retrieval_dir"),
                "exp_01_dir": bundle.get("exp_01_dir"),
                "exp_02_dir": bundle.get("exp_02_dir"),
                "figures": [Path(p).name for p in figures],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return md_path

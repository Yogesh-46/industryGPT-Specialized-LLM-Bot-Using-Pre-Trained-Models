"""Generate thesis figures from executed artefacts only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.paths import ensure_dir, resolve_path

COLOR = "#2F5D50"


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_experiment_figures(
    *,
    bundle: dict[str, Any],
    output_dir: str | Path = "./experiments/results/tables/figures",
    training_log: str | Path = (
        "./experiments/results/adapters/colab_t4_qlora_v1/trainer_log_history.json"
    ),
) -> list[str]:
    """Save PNG charts. Skips quietly if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir = ensure_dir(resolve_path(output_dir))
    saved: list[str] = []

    def _save(fig: Any, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(str(path.as_posix() if hasattr(path, "as_posix") else path))

    history = _load_json(resolve_path(training_log))
    if isinstance(history, list) and history:
        train_steps = [
            (h.get("step"), h.get("loss"))
            for h in history
            if h.get("loss") is not None and h.get("eval_loss") is None
        ]
        eval_pts = [
            (h.get("epoch"), h.get("eval_loss"))
            for h in history
            if h.get("eval_loss") is not None
        ]
        if train_steps:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.plot(
                [s for s, _ in train_steps],
                [v for _, v in train_steps],
                color=COLOR,
                marker="o",
                markersize=3,
            )
            ax.set_title("LoRA-fp16 training loss (colab_t4_qlora_v1)")
            ax.set_xlabel("Step")
            ax.set_ylabel("Train loss")
            _save(fig, "fig_training_loss.png")
        if eval_pts:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(
                [e for e, _ in eval_pts],
                [v for _, v in eval_pts],
                color=COLOR,
                marker="o",
            )
            ax.set_title("Validation loss by epoch (colab_t4_qlora_v1)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Eval loss")
            ax.set_xticks([e for e, _ in eval_pts])
            _save(fig, "fig_eval_loss_epochs.png")

    retrieval = bundle.get("retrieval") or {}
    per_k = retrieval.get("per_top_k") or {}
    if per_k:
        ks = sorted(per_k.keys(), key=lambda x: int(x))
        means = [per_k[k].get("mean_retrieval_relevance_in_domain") or 0.0 for k in ks]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.bar(ks, means, color=COLOR)
        ax.set_ylim(0, 1)
        ax.set_title("Dataset C retrieval relevance vs top-k (in-domain)")
        ax.set_xlabel("top-k")
        ax.set_ylabel("Mean retrieval-relevance proxy")
        _save(fig, "fig_retrieval_topk.png")

        k5 = per_k.get("5") or {}
        cats = k5.get("mean_by_category") or {}
        if cats:
            labels = list(cats.keys())
            fig, ax = plt.subplots(figsize=(8, 4.4))
            ax.bar(labels, [cats[c] or 0.0 for c in labels], color=COLOR)
            ax.set_ylim(0, 1)
            ax.set_title("top-k=5 retrieval relevance by category (in-domain)")
            ax.set_ylabel("Mean retrieval-relevance proxy")
            ax.tick_params(axis="x", rotation=20)
            _save(fig, "fig_retrieval_by_category.png")

        diffs = k5.get("mean_by_difficulty") or {}
        if diffs:
            order = [d for d in ("easy", "medium", "hard") if d in diffs] or list(diffs)
            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            ax.bar(order, [diffs[d] or 0.0 for d in order], color=COLOR)
            ax.set_ylim(0, 1)
            ax.set_title("top-k=5 retrieval relevance by difficulty (in-domain)")
            ax.set_ylabel("Mean retrieval-relevance proxy")
            _save(fig, "fig_retrieval_by_difficulty.png")

    return saved

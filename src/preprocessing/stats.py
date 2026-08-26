"""Dataset statistics and simple visualizations for preprocessing."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def approximate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    if not text:
        return 0
    return max(1, int(round(len(text) / chars_per_token)))


def build_stats(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    *,
    chars_per_token: float = 4.0,
) -> dict[str, Any]:
    """Aggregate preprocessing statistics for academic reporting."""
    by_source = Counter(d["source_id"] for d in accepted)
    by_category = Counter(d["category"] for d in accepted)
    reject_reasons: Counter[str] = Counter()
    for item in rejected:
        for reason in item.get("rejection_reasons", []) or []:
            reject_reasons[reason] += 1
        if item.get("is_duplicate"):
            reject_reasons["duplicate_content"] += 1

    total_chars = sum(int(d.get("char_count", 0)) for d in accepted)
    total_words = sum(int(d.get("word_count", 0)) for d in accepted)
    total_tokens = sum(int(d.get("approx_token_count", 0)) for d in accepted)

    char_counts = [int(d.get("char_count", 0)) for d in accepted]
    return {
        "total_input_documents": len(accepted) + len(rejected),
        "accepted_documents": len(accepted),
        "rejected_documents": len(rejected),
        "total_sources": len(by_source),
        "documents_per_source": dict(by_source),
        "documents_per_category": dict(by_category),
        "total_characters": total_chars,
        "total_words": total_words,
        "total_approx_tokens": total_tokens,
        "chars_per_token_assumption": chars_per_token,
        "duplicate_count": sum(1 for r in rejected if r.get("is_duplicate")),
        "empty_document_count": sum(
            1 for r in rejected if (r.get("validation_flags") or {}).get("empty")
        ),
        "malformed_document_count": sum(
            1 for r in rejected if (r.get("validation_flags") or {}).get("malformed")
        ),
        "rejection_reason_counts": dict(reject_reasons),
        "char_count_summary": {
            "min": min(char_counts) if char_counts else 0,
            "max": max(char_counts) if char_counts else 0,
            "mean": (sum(char_counts) / len(char_counts)) if char_counts else 0.0,
        },
        # Chunk stats belong to Step 7; keep explicit placeholders.
        "final_chunk_count": None,
        "chunks_per_category": None,
        "chunk_stats_status": "Not computed yet (chunking is a later step)",
    }


def write_stats_figures(stats: dict[str, Any], figures_dir: Path) -> list[str]:
    """Write simple bar charts for source/category counts. Returns saved paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return saved

    def _bar(data: dict[str, int], title: str, filename: str, xlabel: str) -> None:
        if not data:
            return
        labels = list(data.keys())
        values = [data[k] for k in labels]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values, color="#2F5D50")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Documents")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        out = figures_dir / filename
        fig.savefig(out, dpi=140)
        plt.close(fig)
        saved.append(str(out))

    _bar(
        stats.get("documents_per_source") or {},
        "Accepted documents per source",
        "documents_per_source.png",
        "Source",
    )
    _bar(
        stats.get("documents_per_category") or {},
        "Accepted documents per category",
        "documents_per_category.png",
        "Category",
    )
    return saved


def write_markdown_report(stats: dict[str, Any], path: Path, figure_paths: list[str]) -> None:
    """Write a short human-readable preprocessing report."""
    lines = [
        "# Preprocessing Report — DataPilot AI",
        "",
        "## Summary",
        "",
        f"- Input documents: **{stats.get('total_input_documents', 0)}**",
        f"- Accepted: **{stats.get('accepted_documents', 0)}**",
        f"- Rejected: **{stats.get('rejected_documents', 0)}**",
        f"- Sources represented: **{stats.get('total_sources', 0)}**",
        f"- Total characters: **{stats.get('total_characters', 0)}**",
        f"- Approx. tokens (chars/{stats.get('chars_per_token_assumption', 4)}): "
        f"**{stats.get('total_approx_tokens', 0)}**",
        f"- Duplicates rejected: **{stats.get('duplicate_count', 0)}**",
        f"- Empty documents: **{stats.get('empty_document_count', 0)}**",
        f"- Malformed documents: **{stats.get('malformed_document_count', 0)}**",
        "",
        "## Documents per source",
        "",
    ]
    for source, count in sorted((stats.get("documents_per_source") or {}).items()):
        lines.append(f"- {source}: {count}")
    lines.extend(["", "## Documents per category", ""])
    for category, count in sorted((stats.get("documents_per_category") or {}).items()):
        lines.append(f"- {category}: {count}")

    lines.extend(
        [
            "",
            "## Chunk statistics",
            "",
            str(stats.get("chunk_stats_status")),
            "",
            "## Figures",
            "",
        ]
    )
    if figure_paths:
        for fp in figure_paths:
            lines.append(f"- `{fp}`")
    else:
        lines.append("- No figures generated (matplotlib unavailable or empty data).")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

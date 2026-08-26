"""Simple rule-based out-of-domain detection for V1."""

from __future__ import annotations

import re

# Lightweight keyword cues — not a learned classifier.
_OOD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "medical",
        re.compile(
            r"\b(diabetes|cancer|diagnose|symptom|prescription|medicine|medical|"
            r"doctor|hospital|treat(ment)?|disease|infection|covid)\b",
            re.I,
        ),
    ),
    (
        "legal",
        re.compile(
            r"\b(lawsuit|attorney|lawyer|criminal|divorce|will and testament|"
            r"legal advice|court case)\b",
            re.I,
        ),
    ),
    (
        "unrelated_personal",
        re.compile(
            r"\b(horoscope|dating advice|relationship advice|astrology|"
            r"lottery numbers)\b",
            re.I,
        ),
    ),
]

_IN_DOMAIN_HINTS = re.compile(
    r"\b(sql|join|cte|select|redshift|postgres|dbt|airflow|etl|elt|"
    r"warehouse|star schema|snowflake schema|scd|power bi|dax|superset|"
    r"dashboard|kpi|cohort|retention|churn|funnel|distkey|sortkey|"
    r"dimension|fact table|pipeline|incremental)\b",
    re.I,
)

OOD_REFUSAL = (
    "I'm DataPilot AI, a Business Intelligence and Data Engineering assistant. "
    "I can help with SQL, data engineering, data warehousing, analytics, and BI "
    "topics, but I can't reliably answer that out-of-domain question."
)


def is_out_of_domain(query: str) -> tuple[bool, str | None]:
    """Return ``(is_ood, category)`` using simple keyword rules.

    If the query also contains strong BI/DE hints, keep it in-domain
    (e.g. \"medical dashboard KPI\" remains allowed).
    """
    text = (query or "").strip()
    if not text:
        return False, None

    has_domain_hint = bool(_IN_DOMAIN_HINTS.search(text))
    for category, pattern in _OOD_PATTERNS:
        if pattern.search(text) and not has_domain_hint:
            return True, category
    return False, None

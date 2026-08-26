# Evaluation data (Dataset C)

Held-out evaluation questions for Systems A/B/C comparison.

**Status:** `eval_set.jsonl` v1.0.0 created (**100** questions).

## Isolation rule

This dataset is **Dataset C** and must remain isolated from Dataset B (fine-tuning).

Do **not** include these `question_id`s or question texts in fine-tuning train/validation splits.

## Files

| File | Purpose |
|------|---------|
| `eval_set.jsonl` | Held-out questions |
| `eval_set_meta.json` | Version + distribution metadata |

## Distribution

| Category | Count |
|----------|------:|
| SQL | 25 |
| Data Engineering | 20 |
| Data Warehousing | 15 |
| BI/Dashboards | 15 |
| Analytics | 15 |
| Out-of-domain | 10 |

Difficulties: easy / medium / hard (mixed within categories).

## Construction notes

- Many SQL / Redshift / Airflow / dbt / Power BI / Superset items cite official documentation URLs from the curated inventory.
- Dimensional modelling and analytics items use established domain concepts and are explicitly marked as such.
- Out-of-domain items evaluate refusal behaviour (not factual domain knowledge).
- No evaluation metrics are fabricated here.

## Commands

```bash
python scripts/prepare_evaluation_data.py
python scripts/validate_evaluation_data.py
```

# Fine-tuning data (Dataset B)

Instruction–response examples for LoRA/QLoRA fine-tuning.

**Status:** `datapilot_ft_v1.1.0` — 667 examples (train 600 / val 67), 0 Dataset C leakage.

## Isolation

Dataset B is separate from Dataset C (`data/evaluation/eval_set.jsonl`).

Validation rejects any instruction that matches an evaluation question (normalized text).

## Files

| File | Purpose |
|------|---------|
| `all.jsonl` | Full Dataset B |
| `train.jsonl` | Training split (~90%) |
| `validation.jsonl` | Validation split (~10%) |
| `dataset_stats.json` | Counts and category distribution |

## Format

```json
{"instruction": "...", "input": "...", "response": "...", "category": "..."}
```

## Commands

```bash
python scripts/prepare_finetuning_data.py
python scripts/validate_finetuning_data.py

# Priority 6 — LoRA (Colab T4: --no-4bit in 2026)
python scripts/train_qlora.py --dry-run
python scripts/train_qlora.py
```

See [docs/finetuning.md](../../docs/finetuning.md) and `notebooks/04_qlora_finetune.ipynb`.

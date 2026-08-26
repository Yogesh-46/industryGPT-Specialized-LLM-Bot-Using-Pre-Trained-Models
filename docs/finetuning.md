# Fine-tuning (QLoRA) — DataPilot AI

**Priority 6.** Train a LoRA adapter on Dataset B for `Qwen/Qwen2.5-1.5B-Instruct`.

## Goals

- Parameter-efficient domain adaptation (response style / BI–DE task behaviour)
- Keep the base model frozen; save adapter separately
- Colab T4–oriented hyperparameters from `config/model.yaml`
- Validation-driven epochs (default **3**, cap **25**, early stopping patience **2**)
- **Never fabricate** train/eval metrics

## Inputs

| Artifact | Path |
|----------|------|
| Train split | `data/finetuning/train.jsonl` |
| Validation split | `data/finetuning/validation.jsonl` |
| Model/LoRA/train config | `config/model.yaml` |

## Outputs

| Artifact | Path |
|----------|------|
| Adapter | `experiments/results/adapters/<run>/adapter/` |
| Summary (real metrics) | `experiments/results/adapters/<run>/training_summary.json` |

## Commands

```bash
# No GPU: validate Dataset B formatting
python scripts/train_qlora.py --dry-run

# Full QLoRA (requires NVIDIA CUDA; Colab T4 recommended)
python scripts/train_qlora.py

# Tiny GPU smoke run
python scripts/train_qlora.py --max-train-samples 32 --max-val-samples 8 --run-name smoke_qlora
```

Colab: `notebooks/04_qlora_finetune.ipynb`

## Colab troubleshooting

Current Colab (2026) ships **PyTorch 2.11 + CUDA 12.8**. bitsandbytes has no
`libbitsandbytes_cuda128.so`, falls back to a **CPU** wheel, then crashes:

`ModuleNotFoundError: No module named 'triton.ops'`

`transformers` can still import leftover `bitsandbytes`, which then imports removed `triton.ops`.
The trainer now installs a compatibility stub **before** Hugging Face imports.

In Colab, also uninstall bitsandbytes, then **Restart session**:

```python
%pip uninstall -y bitsandbytes
```

Copy updated `src/finetuning/train.py` and `scripts/train_qlora.py`, then:

```bash
!python scripts/train_qlora.py --run-name colab_t4_qlora_v1 --no-4bit -v
```

4-bit QLoRA remains the config default for machines with a working GPU bitsandbytes
build. Record `method` from `training_summary.json` (`LoRA-fp16` vs `QLoRA-4bit`).

### `TypeError: ... unexpected keyword argument 'warmup_ratio'`

Colab’s Transformers v5 removed `warmup_ratio` (use `warmup_steps` as a float ratio).
Copy the updated `src/finetuning/train.py` to Drive; the trainer now detects the installed API.

### `TypeError: SFTTrainer ... unexpected keyword argument 'dataset_text_field'` / `'tokenizer'`

Current TRL moved those fields into `SFTConfig` and uses `processing_class`. Copy the same `train.py` again.

### `ImportError: incompatible version of torchao ... 0.10.0`

PEFT requires torchao > 0.16; Colab often has 0.10. Standard LoRA does not need it.
The trainer now skips an incompatible torchao install. Optionally:

```python
%pip uninstall -y torchao
```

## Hyperparameters (V1 defaults)

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Quantisation (config default) | 4-bit NF4 double-quant |
| Quantisation (executed Colab) | fp16 (no bitsandbytes CUDA 12.8 wheel) |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| Batch × accum | 1 × 8 |
| LR | 2e-4 cosine |
| Max seq length | 1024 |
| Optim | paged_adamw_8bit |

## Executed V1 run (`colab_t4_qlora_v1`)

| Field | Value |
|-------|-------|
| Method | **LoRA-fp16** (`use_4bit: false`) |
| GPU | Tesla T4 |
| Train / val | 600 / 67 |
| Epochs | 3 |
| Best eval loss | 0.595 |
| Wall clock | ~977 s |
| Peak GPU memory | ~4.56 GiB |

See `experiments/results/adapters/colab_t4_qlora_v1/training_summary.json`. Adapter quality claims use Priority 9–10 tables, not training loss alone.

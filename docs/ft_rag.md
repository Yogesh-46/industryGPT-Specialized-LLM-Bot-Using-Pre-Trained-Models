# Fine-tuned + RAG (System C) — DataPilot AI

**Priority 7.** Attach the trained LoRA adapter to the existing RAG chatbot.

## What System C is

Same pipeline as System B (retrieve → prompt → generate → sources), with
`PeftModel.from_pretrained` on `Qwen/Qwen2.5-1.5B-Instruct`.

| System | RAG | Adapter |
|--------|-----|---------|
| A `baseline_llm` | No | No |
| B `rag_only` | Yes | No |
| C `finetuned_rag` | Yes | Yes (`colab_t4_qlora_v1`) |

Training used **LoRA-fp16** on Colab T4 (`use_4bit: false`). Inference follows the
same fallback if 4-bit bitsandbytes is unavailable.

## Adapter location

Configured in `config/model.yaml`:

```yaml
paths:
  adapter_path: ./experiments/results/adapters/colab_t4_qlora_v1/adapter
```

Required files: `adapter_config.json` + `adapter_model.safetensors`.

## Commands

```bash
# System C CLI
python scripts/run_finetuned_rag.py --retrieve-only "What is DISTKEY?"
python scripts/run_finetuned_rag.py --interactive

# Same thing via baseline script
python scripts/run_baseline_rag.py --finetuned-rag "Explain star schema."

# Framework smoke (not academic scores)
python scripts/evaluate.py --systems finetuned_rag --limit 5 --mock-llm

# Real System C eval (loads LLM + adapter; GPU recommended)
python scripts/evaluate.py --systems finetuned_rag --limit 10
```

Executed A/B/C automatic metrics: [docs/evaluation.md](evaluation.md) and `experiments/results/tables/results.md`. Do not cite mock-LLM or bitsandbytes `generation_error` runs.

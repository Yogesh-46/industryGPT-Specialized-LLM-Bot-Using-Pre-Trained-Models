# Model Selection — DataPilot AI

**Decision date:** 2026-08-18  
**Status:** FINAL for V1 (single model; no multi-model training)  
**Selected model:** `Qwen/Qwen2.5-1.5B-Instruct`

This document records the Priority 4 decision for academic traceability.  
It does **not** invent latency/accuracy benchmarks. Selection is based on explicit feasibility criteria for Google Colab T4 + QLoRA + RAG.

---

## Selection criteria (priority order)

1. **Colab T4 compatibility** (~15GB VRAM) for QLoRA training and RAG inference  
2. **Inference practicality** for demo and evaluation loops  
3. **PEFT / LoRA / QLoRA compatibility**  
4. **Instruction-tuned behaviour** (chat/instruct checkpoint)  
5. **Model quality ceiling** (secondary to feasibility)

Constraint from project scope: **fine-tune exactly one model**.

---

## Candidates compared

| Candidate | HF id | ~Params | 4-bit / PEFT | T4 practicality | Access friction | Decision |
|-----------|-------|--------:|--------------|-----------------|-----------------|----------|
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Yes | Excellent (headroom for RAG+adapter) | Low | **SELECTED** |
| Gemma-2-2B-IT | `google/gemma-2-2b-it` | 2B | Yes | Good | May require HF license acceptance | Rejected for access friction |
| Phi-3.5-mini-Instruct | `microsoft/Phi-3.5-mini-instruct` | 3.8B | Yes | Moderate (tighter with long RAG context) | Low–moderate | Rejected as heavier than needed |
| Mistral-7B-Instruct | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Yes | Tight on T4 with QLoRA + long context | Low–moderate | Rejected for VRAM risk |

---

## Why Qwen2.5-1.5B-Instruct

1. **Best T4 safety margin** for the hybrid setup: embedding model + FAISS retrieval + 4-bit LLM + LoRA adapter.  
2. **Instruction-tuned** checkpoint suitable for grounded RAG prompting.  
3. **PEFT-friendly** architecture (`q/k/v/o` and MLP projections already listed in `config/model.yaml`).  
4. **Faster iteration** for evaluation (Systems A/B/C) than 3.8B–7B options on limited academic compute.  
5. Quality risk is mitigated by **RAG grounding** (primary factual pathway), so extreme parameter count is not required for V1.

## Why not the alternatives

- **Gemma-2-2B:** comparable size, but gated access can block reproducible Colab setup.  
- **Phi-3.5-mini (~3.8B):** stronger reasoning potential, but less headroom for long retrieved contexts on T4.  
- **Mistral-7B:** higher quality ceiling, but highest risk of training/inference instability on T4 within project time.

---

## Locked configuration

Source of truth: `config/model.yaml`

- `selected.model_id`: `Qwen/Qwen2.5-1.5B-Instruct`  
- `selected.selection_status`: `final_v1`  
- Quantisation: 4-bit NF4 (when CUDA available)  
- Fine-tuning method: QLoRA / LoRA via PEFT  
- Adapter output: `experiments/results/adapters/`

---

## What this decision is not

- Not a claim that Qwen is the globally best BI model  
- Not based on fabricated win-rates vs other candidates  
- Not a commitment to multi-model ensembling  

Final academic claims about answer quality come from **executed** evaluation runs (`experiments/results/tables/results.md`), not from this selection memo.

---

## Executed training / inference (after this decision)

Colab T4 in 2026 could not run 4-bit QLoRA (bitsandbytes missing `libbitsandbytes_cuda128.so`). The locked **model id** did not change. Training and Systems A/B/C evaluation used **LoRA-fp16** / fp16 generation (`--no-4bit` / `load_in_4bit: false` on Drive).

---

## Implications for later phases

- Dataset B: `data/finetuning/` (667 examples; isolated from Dataset C)  
- Training: `scripts/train_qlora.py` / `notebooks/04_qlora_finetune.ipynb`  
- FT+RAG: `scripts/run_finetuned_rag.py` (adapter `colab_t4_qlora_v1`)  
- Streamlit + experiments: `scripts/run_app.py`, `notebooks/05_experiments.ipynb`, `notebooks/06_evaluation.ipynb`

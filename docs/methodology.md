# Methodology — DataPilot AI

## Research framing

This project implements an industry-specific conversational AI for BI and Data Engineering, combining:

1. Curated domain documentation (Dataset A)  
2. Instruction fine-tuning with LoRA/QLoRA (Dataset B)  
3. Held-out evaluation (Dataset C)  
4. Retrieval-Augmented Generation  
5. A Streamlit demonstration interface  

The methodology is designed so artefacts (configs, logs, datasets, metrics) can feed a subsequent research paper (introduction, industry analysis, literature, methods, results, discussion, conclusion).

---

## High-level method

1. Curate official documentation URLs (no indiscriminate scraping)  
2. Collect and preprocess documents with provenance  
3. Chunk, embed, and index with FAISS  
4. Build a baseline RAG chatbot  
5. Select a Colab T4–compatible Hugging Face LLM  
6. Prepare and validate a fine-tuning instruction set  
7. Train with LoRA/QLoRA; save adapter separately  
8. Combine fine-tuned model with RAG  
9. Evaluate baseline vs RAG vs fine-tuned+RAG  
10. Document limitations and reproducible settings  

---

## Model selection criteria

Candidates were compared for Google Colab T4 feasibility, 4-bit quantisation, PEFT/QLoRA compatibility, instruction-tuned behaviour, and evaluation iteration speed.

**Final V1 model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Status:** `final_v1` (locked in `config/model.yaml`)  
**Rationale document:** [`docs/model_selection.md`](model_selection.md)

This decision prioritises reproducible T4 training/inference for a hybrid RAG + PEFT system. It is **not** a fabricated claim that Qwen is globally superior to all alternatives. Answer-quality claims use executed evaluation tables (`experiments/results/tables/results.md`).

---

## Fine-tuning policy

- Use LoRA or QLoRA with PEFT  
- Small batches + gradient accumulation for T4  
- Mixed precision and gradient checkpointing as needed  
- **Do not default to 25 epochs**; choose based on validation behaviour and overfitting risk (cap documented as 25 for university constraints)  
- Record training/validation loss, runtime, GPU memory where possible  
- Never fabricate training curves or metrics  
- Implementation: [`docs/finetuning.md`](finetuning.md), `scripts/train_qlora.py`, `notebooks/04_qlora_finetune.ipynb`  

---

## RAG policy

- Configurable chunk size, overlap, separators, top-k  
- Lightweight Sentence Transformers embeddings (configurable)  
- FAISS with persistent index and metadata mapping  
- System prompt enforces scope, grounding, and citation honesty  
- Optional reranking only if evaluation justifies complexity  

Starting chunk settings (`config/rag.yaml`): ~650 tokens, ~75 overlap — **not claimed optimal**.

---

## Prompting & safety of claims

Prompts live in `config/prompts/`, not hard-coded in business logic. The assistant must:

- Stay in BI/DE scope  
- Prefer retrieved context for factual answers  
- Avoid unsupported claims and fabricated citations  
- Admit uncertainty  
- Refuse out-of-domain queries politely  
- Not claim SQL was executed  

---

## Reproducibility

- Random seeds in configs  
- `requirements.txt` version ranges  
- `.env.example` without secrets  
- Experiment JSON/CSV logs  
- Rebuildable vector store from processed corpus  
- Separate adapter artefacts  

---

## Integrity

- No fabricated datasets, metrics, citations, or performance claims  
- Dataset C isolated from training  
- Optional experiments and human grades stay uncollected rather than invented  

---

## Current implementation stage

Priorities 1–12 complete for V1: Dataset A/B/C, FAISS RAG, LoRA-fp16 adapter (`colab_t4_qlora_v1`), Streamlit UI, executed exp_01/exp_02/exp_05 tables, and Colab notebooks 01–06.

**Executed compute note:** Colab T4 (PyTorch 2.11, CUDA 12.8) could not load bitsandbytes GPU kernels. Training and LLM evaluation used **fp16 LoRA**, not 4-bit QLoRA. Config still documents 4-bit as the intended default when a compatible bitsandbytes build exists.

Optional exp_03/04/06 and human 1–5 ratings remain uncollected. Do not invent those scores.

# Architecture — DataPilot AI

## Overview

DataPilot AI uses a **hybrid, non-agentic** architecture:

1. **RAG** over a curated BI/Data Engineering knowledge corpus  
2. **LoRA/QLoRA** fine-tuning of a Hugging Face open-source LLM  
3. A **Streamlit** chat UI with session memory and source attribution  

V1 does **not** include autonomous tool use, live SQL execution against production systems, multi-agent orchestration, or automated dashboard modification.

## Component diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Chat UI                       │
│  title · examples · history · sources · loading / errors    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────┐
│                   Query Processing                          │
│  normalize · bound conversation memory · scope checks       │
└─────────────────────────────┬───────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              v                               v
┌──────────────────────────┐    ┌─────────────────────────────┐
│     Embedding Encoder    │    │   Prompt Builder            │
│  (Sentence Transformers) │    │  system + history + context │
└─────────────┬────────────┘    └──────────────▲──────────────┘
              │                                │
              v                                │
┌──────────────────────────┐                   │
│   FAISS Vector Store     │── top-k chunks ───┘
│   + metadata / URLs      │
└──────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────┐
│         Hugging Face LLM (± LoRA/QLoRA adapter)             │
│         grounded generation + uncertainty handling          │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              v
                 Answer + Source Attribution
```

## Data assets (strict separation)

| Asset | Role |
|-------|------|
| Dataset A | RAG corpus (documents → chunks → embeddings → FAISS) |
| Dataset B | Instruction fine-tuning (train/validation only) |
| Dataset C | Held-out evaluation questions |

## Configuration surfaces

| File | Responsibility |
|------|----------------|
| `config/model.yaml` | Base model candidates, quantisation, LoRA, training, inference |
| `config/rag.yaml` | Embeddings, chunking, retrieval, memory, caching |
| `config/sources.yaml` | Curated URL inventory and provenance policy |
| `config/evaluation.yaml` | Systems under test, experiments, metrics |
| `config/prompts/system_prompt.txt` | Domain scope and behavioural rules |

## Runtime modes

| Mode | RAG | Fine-tuned adapter | Use |
|------|-----|--------------------|-----|
| `baseline_llm` | No | No | Experiment 1 baseline |
| `rag_only` | Yes | No | System B |
| `finetuned_rag` | Yes | Yes (`colab_t4_qlora_v1`) | System C |

CLI: `python scripts/run_finetuned_rag.py` — see [docs/ft_rag.md](ft_rag.md).

## Design principles

- Configuration over hard-coding  
- Provenance on every chunk  
- Restartable, cacheable pipelines  
- Colab T4 feasibility  
- No fabricated metrics or citations  
- Simple solutions preferred over unnecessary frameworks  

## Status

Priorities 1–12 complete for V1 (including notebooks and docs aligned with executed LoRA-fp16 Colab runs). Streamlit: [docs/app.md](app.md). Results: [docs/evaluation.md](evaluation.md).

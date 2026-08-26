# DataPilot AI

**An AI Copilot for Business Intelligence & Data Engineering**

> Master's **capstone** (IndustryGPT): industry-specific LLM bot using a Hugging Face model, domain data, RAG, and LoRA on Google Colab T4.  
> Submission notebook: [`notebooks/00_capstone_submission.ipynb`](notebooks/00_capstone_submission.ipynb) (same layout as the course Colab template: Project Summary + GitHub link + reasoning + runnable sections).

**GitHub:** https://github.com/Yogesh-46/DataPilot-AI *(replace if your public repo name differs)*


---

## 1. Project overview

DataPilot AI is a domain-specific conversational assistant that helps BI developers, data engineers, analysts, and students with SQL, data engineering, warehousing, dashboards, BI platforms, and analytics concepts.

Unlike a generic chatbot, DataPilot AI combines:

1. **Retrieval-Augmented Generation (RAG)** over a curated BI Knowledge Hub  
2. **Parameter-efficient fine-tuning (LoRA/QLoRA)** on domain instruction data  
3. An open-source **Hugging Face** instruction-tuned LLM suitable for Google Colab T4  

V1 is **not agentic**: no autonomous database execution, dashboard modification, or multi-agent orchestration.

---

## 2. Problem statement

BI and data engineering practitioners often need grounded answers about SQL dialects, warehouse design, orchestration tools, and BI platforms. Generic LLMs can be fluent but may hallucinate syntax, invent product behaviour, or miss domain nuance.

**Working research question:**  
*Can a domain-specific AI Copilot built using Large Language Models and Retrieval-Augmented Generation improve the accuracy and contextual relevance of Business Intelligence and Data Engineering support compared with a generic conversational AI?*

**Working hypothesis:**  
*A conversational AI enhanced with BI-specific knowledge through RAG (and optionally domain fine-tuning) will produce more accurate, context-aware, and reliable responses than a generic LLM.*

These statements may be refined after literature review and experimentation.

---

## 3. Industry / domain

| Item | Detail |
|------|--------|
| Industry | Technology & Information Technology |
| Domain | Business Intelligence and Data Engineering |
| Users | BI developers, data engineers, analysts, analytics engineers, junior SQL developers, BI students |

---

## 4. Research motivation

This implementation supports a research paper extending Deep Learning for NLP work. Draft: [docs/paper/research_paper.md](docs/paper/research_paper.md). Methodology, experiments, results, and limitations are traceable from this repository.

---

## 5. Architecture

Hybrid **RAG + LoRA/QLoRA + Hugging Face LLM** (non-agentic):

```
User → Streamlit Chat → Query Processing → Retriever → FAISS Vector DB
                              ↓
                    Relevant Domain Context
                              ↓
                    Fine-Tuned HF LLM → Grounded Response + Source Attribution
```

See [docs/architecture.md](docs/architecture.md).

---

## 6. Knowledge sources

Curated official documentation inventory (not indiscriminate scraping):

- PostgreSQL Documentation  
- Amazon Redshift Documentation  
- Microsoft Learn / Power BI Documentation  
- Apache Superset Documentation  
- Apache Airflow Documentation  
- dbt Documentation  

Full inventory: [config/sources.yaml](config/sources.yaml) and [docs/dataset.md](docs/dataset.md).

---

## 7. Dataset construction

Three **separate** assets:

| Asset | Purpose | Location |
|-------|---------|----------|
| **Dataset A** | RAG knowledge corpus | `data/raw`, `data/processed`, `knowledge_base/` |
| **Dataset B** | Fine-tuning instructions (667 examples; train/val) | `data/finetuning/` |
| **Dataset C** | Held-out evaluation (~100 questions) | `data/evaluation/` |

Evaluation data must never leak into training. Dataset B (667) and Dataset C (100) are already created and isolated.

---

## 8. Data preprocessing

Executed pipeline: HTML cleanup, nav/boilerplate removal, whitespace normalisation, duplicate/empty/malformed detection, length and metadata validation, statistics + EDA (`notebooks/03_eda.ipynb`). **37** accepted documents from **42** downloads (5 exact-page duplicates removed).

Details: [docs/dataset.md](docs/dataset.md), [docs/methodology.md](docs/methodology.md).

---

## 9. Model selection

**Final V1 model:** `Qwen/Qwen2.5-1.5B-Instruct`. Config still lists 4-bit QLoRA as the default; the **executed** Colab T4 training and evaluation used **LoRA-fp16** because bitsandbytes has no CUDA 12.8 wheel (2026 Colab).

Rationale and rejected alternatives: [docs/model_selection.md](docs/model_selection.md). Config: [config/model.yaml](config/model.yaml).

---

## 10. Fine-tuning

LoRA (PEFT) via `scripts/train_qlora.py`; 4-bit QLoRA when bitsandbytes works. Epoch count is **validation-driven** (not defaulted to 25). Adapter saved separately from the base model.

Training entrypoints: `scripts/train_qlora.py`, Colab notebook `notebooks/04_qlora_finetune.ipynb`. Details: [docs/finetuning.md](docs/finetuning.md).

Status: **adapter trained** (`colab_t4_qlora_v1`, LoRA-fp16). FT+RAG: [docs/ft_rag.md](docs/ft_rag.md).

---

## 11. RAG

Configurable chunking (~650 tokens, ~75 overlap as a starting point), Sentence Transformers embeddings, FAISS index with provenance metadata, prompt construction with source attribution.

Config: [config/rag.yaml](config/rag.yaml).

---

## 12. Evaluation

Compare at minimum:

1. Baseline LLM (no domain RAG)  
2. RAG-only DataPilot AI  
3. Fine-tuned + RAG (if compute allows; otherwise document limitation)

Automatic metrics (executed): point coverage, token F1, ROUGE-L, groundedness/retrieval proxies, latency. Human 1–5 rubric is a blank template only.

**[Results]** — training, retrieval, and LLM A/B/C tables: [experiments/results/tables/results.md](experiments/results/tables/results.md). See [docs/evaluation.md](docs/evaluation.md).

---

## 13. Installation

```bash
# Clone / open repository
cd Masters_Project   # or your project root

# Create virtual environment (recommended)
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Environment
copy .env.example .env
# Edit .env — add HF_TOKEN only if needed for gated models
```

Google Colab: T4 GPU for `notebooks/04_qlora_finetune.ipynb` and `notebooks/05_experiments.ipynb`. Collection/EDA/results notebooks (01–03, 06) do not need a GPU.

---

## 14. Usage

```bash
python scripts/collect_documents.py
python scripts/preprocess_documents.py
python scripts/build_vector_store.py
python scripts/run_baseline_rag.py --retrieve-only
python scripts/train_qlora.py --dry-run
# GPU (Colab T4): python scripts/train_qlora.py --no-4bit
python scripts/run_finetuned_rag.py --retrieve-only "What is DISTKEY?"
python scripts/evaluate.py --systems finetuned_rag --limit 5 --mock-llm
python scripts/run_app.py
python scripts/run_retrieval_experiment.py
python scripts/build_results_tables.py
# GPU (Colab T4): python scripts/run_experiments.py --exp exp_01,exp_02 --no-4bit
```

**Current status:** Priorities 1–12 complete for V1 (pipelines, adapter, Streamlit, executed A/B/C tables, Colab notebooks, documentation aligned with LoRA-fp16 Colab runs). Optional exp_03/04/06 and human 1–5 ratings are not collected.

---

## 15. Example questions

- Explain Slowly Changing Dimension Type 2.  
- Why is my LEFT JOIN creating duplicate rows?  
- Write a Redshift query for monthly active users.  
- How should I design a retention dashboard?  
- What is the difference between DISTKEY and SORTKEY?  

Out-of-domain example: medical diagnosis questions should be refused politely.

---

## 16. Limitations

- V1 is conversational + RAG (+ optional PEFT); not a live query engine or BI platform integration.  
- Knowledge coverage is limited to the curated inventory.  
- Colab T4 (2026) has no bitsandbytes CUDA 12.8 wheel; V1 training and eval used **fp16 LoRA**, not 4-bit QLoRA.  
- Keyword OOD filter refused 5/100 eval items (Dataset C has 10 OOD questions).  
- Automatic metrics do not fully replace expert human review.  
- Experiment metrics: [experiments/results/tables/results.md](experiments/results/tables/results.md). Automatic scores are lexical proxies, not human grades.

---

## 17. Future work

- Agentic tooling (optional)  
- Live warehouse/SQL execution sandboxes  
- Broader dialect and platform coverage  
- Stronger reranking and citation verification  
- Larger, expert-validated instruction sets  

---

## 18. Repository structure

```
Masters_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── config/
│   ├── model.yaml
│   ├── rag.yaml
│   ├── sources.yaml
│   ├── evaluation.yaml
│   └── prompts/system_prompt.txt
├── data/                  # raw, processed, finetuning, evaluation
├── knowledge_base/        # documents, chunks, vector_store, source_inventory
├── src/                   # ingestion, preprocessing, embeddings, retrieval, ...
├── app/                   # Streamlit chat UI
├── notebooks/             # 01 collection, 02 preprocess, 03 EDA, 04 QLoRA, 05 GPU experiments, 06 results
├── scripts/               # CLI pipelines
├── experiments/           # logs + results
├── docs/                  # architecture, dataset, evaluation, methodology
└── tests/
```

---

## Academic assessment mapping

| Requirement | How this repo supports it |
|-------------|---------------------------|
| Industry-relevant dataset | Curated BI/DE official docs + domain instruction/eval sets |
| Data quality | Preprocessing + validation + statistics (`data/processed/stats/`, `notebooks/03_eda.ipynb`) |
| Innovative techniques | Hybrid RAG + PEFT LoRA (fp16 on Colab T4; QLoRA when bitsandbytes works) |
| Fine-tuning | PEFT LoRA (`scripts/train_qlora.py`); adapter `colab_t4_qlora_v1` |
| Bot functionality | Streamlit chat (`python scripts/run_app.py`) + evaluation harness |
| Documentation | README + `docs/` + experiment logs |

---

## Integrity rules

- Do not fabricate datasets, metrics, citations, or training results.  
- Prefer official documentation and preserve provenance.  
- Keep Dataset C isolated from training.  
- Record real experiment outcomes only.

---

## License / attribution

Documentation content from third parties remains under their respective licenses and terms. This academic project stores provenance and attributes sources in RAG responses.

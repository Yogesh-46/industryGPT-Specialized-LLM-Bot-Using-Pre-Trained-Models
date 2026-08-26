# Evaluation Plan — DataPilot AI

## Goal

Compare systems fairly and reproducibly to address the working research question on domain-specific RAG (± fine-tuning) versus a generic LLM for BI/Data Engineering support.

**No fabricated scores.** Use `[To be populated after experiment]` only for experiments that have not been executed.

---

## Systems under test

| ID | Description | RAG | Fine-tune |
|----|-------------|-----|-----------|
| `baseline_llm` | Base/pre-trained LLM | No | No |
| `rag_only` | RAG DataPilot AI | Yes | No |
| `finetuned_rag` | Fine-tuned + RAG | Yes | Yes |

If compute prevents a comparison, document the limitation rather than inventing results.

---

## Experiments

| ID | Name | Required |
|----|------|----------|
| `exp_01_baseline_vs_rag` | Generic/base LLM vs RAG | Yes |
| `exp_02_rag_vs_finetuned_rag` | RAG vs Fine-tuned + RAG | Yes |
| `exp_03_embedding_comparison` | Embedding model comparison | Optional |
| `exp_04_chunk_size_comparison` | Chunk size comparison | Optional |
| `exp_05_topk_comparison` | Top-k comparison | Optional |
| `exp_06_latency_comparison` | Latency comparison | Optional |

Every run must log: experiment ID, datetime, model, dataset version, configuration, parameters, runtime, metrics, output location.

---

## Automatic metrics (implemented)

Lexical / overlap proxies (not human grades):

- Point coverage, token F1, ROUGE-L  
- Groundedness / retrieval-relevance proxies (RAG)  
- OOD refusal cue score  
- Response latency  

Optional: embedding cosine similarity (`--semantic-similarity`). BERTScore is listed as a dependency if used later; V1 tables use the metrics above.

Configuration: `config/evaluation.yaml`.

---

## Human evaluation (if feasible)

Scale **1–5** for:

- Correctness  
- Relevance  
- Completeness  
- Groundedness  

**Do not fabricate human scores.**

---

## Evaluation dataset

Held-out Dataset C: `data/evaluation/eval_set.jsonl` (**v1.0.0**, **100** questions).

Status: **created**. Isolation from fine-tuning is mandatory.

Difficulty mix: easy 32 / medium 46 / hard 22.

---

## Results

Canonical file (regenerated from artefacts): [`experiments/results/tables/results.md`](../experiments/results/tables/results.md)

| Experiment | Status | Key metrics |
|------------|--------|-------------|
| Fine-tuning `colab_t4_qlora_v1` | Completed (Colab T4) | LoRA-fp16; best eval loss **0.595**; ~977 s; 4.56 GiB |
| exp_05 retrieval top-k | Completed (no LLM) | In-domain mean relevance k=3/5/8: **0.633 / 0.733 / 0.794** (n=90) |
| exp_01 baseline vs RAG | Completed (Colab T4, n=100, fp16) | Point coverage **0.470** vs **0.510**; RAG slower (~9.1 s vs ~5.4 s) |
| exp_02 RAG vs FT+RAG | Completed (Colab T4, n=100, fp16) | Point coverage **0.505** vs **0.430**; FT+RAG higher token F1 **0.333** / ROUGE-L **0.282** |
| Framework smoke (`--mock-llm`) | Validation only | **Not** model performance |
| exp_03, exp_04, exp_06 | Optional / not run | `[To be populated after experiment]` |

### Table 3 excerpt (executed)

| System | n | Point coverage | Token F1 | ROUGE-L | Latency (ms) |
|--------|--:|---------------:|---------:|--------:|-------------:|
| `baseline_llm` | 100 | 0.470 | 0.162 | 0.124 | 5435 |
| `rag_only` | 100 | 0.510 | 0.129 | 0.104 | 9098 |

### Table 4 excerpt (executed)

| System | n | Point coverage | Token F1 | ROUGE-L | Latency (ms) |
|--------|--:|---------------:|---------:|--------:|-------------:|
| `rag_only` | 100 | 0.505 | 0.134 | 0.108 | 8094 |
| `finetuned_rag` | 100 | 0.430 | 0.333 | 0.282 | 3564 |

These are **lexical proxies** on held-out Dataset C. They are not human 1–5 grades. Inference was fp16 on Tesla T4 (`load_in_4bit: false`). Keyword OOD refusal fired on 5/100 items (Dataset C has 10 OOD questions).

Do not cite the earlier bitsandbytes `generation_error` folders (point coverage ≈ 0). Canonical run ids: `exp_01_baseline_vs_rag_20260820T141231Z`, `exp_02_rag_vs_finetuned_rag_20260820T143647Z`.

### Table 2 excerpt (executed)

| top-k | In-domain n | Mean retrieval relevance |
|------:|------------:|-------------------------:|
| 3 | 90 | 0.633 |
| 5 | 90 | 0.733 |
| 8 | 90 | 0.794 |

At k=5, SQL 0.860 · DE 0.825 · BI 0.800 · DW 0.667 · Analytics 0.400.

Difficulty (k=5, in-domain): easy 0.852 · medium 0.738 · hard 0.571.

At k=5, **12** in-domain items had zero lexical coverage (`failure_cases.md` next to the exp_05 run). Analytics is the weakest category (0.400).

This is a **lexical proxy** of expected-answer-point coverage in retrieved chunks. It is not generation quality.

Figures (from executed training + retrieval): `experiments/results/tables/figures/`.

### Human evaluation

Blank rubric only: `experiments/results/human_eval/rubric_blank.csv` (1–5 on correctness, relevance, completeness, groundedness). **Do not invent ratings.** Fill after reading real system answers.

### How to regenerate tables

```bash
python scripts/build_results_tables.py
```

Notebooks: `01` collection, `02` preprocess, `03` EDA, `04` training, `05` GPU experiments, `06` results interpretation.

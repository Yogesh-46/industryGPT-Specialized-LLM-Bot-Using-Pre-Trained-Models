# DataPilot AI — experiment results

Numbers below come only from executed runs. Tables 3-4 are from executed Colab T4 runs (fp16 inference; not mock LLM).

## Table 1 - Fine-tuning (executed, Colab T4)

| Field | Value |
|-------|-------|
| Run | `colab_t4_qlora_v1` |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | LoRA-fp16 |
| GPU | Tesla T4 |
| Train / val examples | 600 / 67 |
| Epochs configured | 3.0 |
| Wall clock (s) | 977.3 |
| Peak GPU memory (GiB) | 4.558 |
| Train loss (Trainer) | 0.8251 |
| Best eval loss | 0.5951 |
| Eval loss by epoch | [1.0101398229599, 0.67005854845047, 0.5950806736946106] |

## Table 2 - Retrieval on Dataset C (executed, no LLM)

Dataset: 100 held-out questions. Metric: mean retrieval-relevance proxy on **in-domain** items.

| top-k | In-domain n | Mean retrieval relevance | Zero-coverage n | Median retrieve latency (ms) |
|------:|------------:|-------------------------:|----------------:|-----------------------------:|
| 3 | 90 | 0.633 | 19 | 75.8 |
| 5 | 90 | 0.733 | 12 | 75.8 |
| 8 | 90 | 0.794 | 9 | 75.8 |

### Table 2b - top-k=5 by category (in-domain)

| Category | Mean retrieval relevance |
|----------|-------------------------:|
| Analytics | 0.400 |
| BI/Dashboards | 0.800 |
| Data Engineering | 0.825 |
| Data Warehousing | 0.667 |
| SQL | 0.860 |

### Table 2c - top-k=5 by difficulty (in-domain)

| Difficulty | Mean retrieval relevance |
|------------|-------------------------:|
| easy | 0.852 |
| medium | 0.738 |
| hard | 0.571 |

### Table 2d - Out-of-domain retrieval (not in in-domain means)

OOD n=10; mean retrieval-relevance proxy 0.100. OOD items should still be refused by the chatbot even if chunks are retrieved.


## Table 3 - exp_01: baseline LLM vs RAG (executed, Colab T4)

| System | n | Mean point coverage | Mean token F1 | Mean ROUGE-L | Mean latency (ms) |
|--------|--:|--------------------:|--------------:|-------------:|------------------:|
| `baseline_llm` | 100 | 0.470 | 0.162 | 0.124 | 5435.3 |
| `rag_only` | 100 | 0.510 | 0.129 | 0.104 | 9098.0 |

## Table 4 - exp_02: RAG vs fine-tuned + RAG (executed, Colab T4)

| System | n | Mean point coverage | Mean token F1 | Mean ROUGE-L | Mean latency (ms) |
|--------|--:|--------------------:|--------------:|-------------:|------------------:|
| `rag_only` | 100 | 0.505 | 0.134 | 0.108 | 8093.7 |
| `finetuned_rag` | 100 | 0.430 | 0.333 | 0.282 | 3564.2 |

## Figures

- `fig_training_loss.png` -- `experiments/results/tables/figures/fig_training_loss.png`
- `fig_eval_loss_epochs.png` -- `experiments/results/tables/figures/fig_eval_loss_epochs.png`
- `fig_retrieval_topk.png` -- `experiments/results/tables/figures/fig_retrieval_topk.png`
- `fig_retrieval_by_category.png` -- `experiments/results/tables/figures/fig_retrieval_by_category.png`
- `fig_retrieval_by_difficulty.png` -- `experiments/results/tables/figures/fig_retrieval_by_difficulty.png`

## Integrity

- Mock LLM smoke tests are **not** copied into Tables 3-4.
- Human 1-5 ratings are not fabricated. Blank rubric: `experiments/results/human_eval/rubric_blank.csv`.
- Retrieval failure cases (zero coverage at k=5) are listed next to the exp_05 run folder when artefacts exist.
- Inference for Tables 3-4 used fp16 on Tesla T4 (`load_in_4bit: false`) after bitsandbytes was unavailable.
- Keyword OOD filter refused 5/100 items; Dataset C has 10 Out-of-domain questions.


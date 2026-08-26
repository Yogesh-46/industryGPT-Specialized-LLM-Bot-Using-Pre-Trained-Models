# DataPilot AI: A Domain-Specific Conversational Copilot for Business Intelligence and Data Engineering Using RAG and Parameter-Efficient Fine-Tuning

**Author:** [Your Name]  
Master's in CS: Artificial Intelligence and Machine Learning, Woolf University

---

## Abstract

Generic instruction-tuned language models are fluent on Business Intelligence (BI) and Data Engineering (DE) topics but routinely invent product behaviour, omit operational constraints, or fail to cite sources. This paper presents **DataPilot AI**, a non-agentic industry chatbot that combines retrieval-augmented generation (RAG) over curated official documentation with parameter-efficient fine-tuning of a small open-source model. The system is implemented under Google Colab Tesla T4 constraints. The locked model is Qwen2.5-1.5B-Instruct. Four-bit QLoRA was planned; Colab’s PyTorch 2.11 / CUDA 12.8 environment had no working bitsandbytes GPU wheel, so **LoRA in fp16** was used (peak allocated memory 4.56 GiB; training wall-clock 977 s; best validation loss 0.595). Three systems were compared on a held-out set of 100 questions isolated from training: (A) base LLM, (B) RAG only, (C) LoRA adapter plus RAG. Automatic lexical proxies—not human grades—show a small RAG gain in expected-answer-point coverage (0.470 to 0.510) at higher latency. Fine-tuned RAG lowered point coverage (0.430) but raised token F1 (0.333 versus 0.134) and ROUGE-L (0.282 versus 0.108). Retrieval relevance at top-k=5 was 0.733 in-domain, with Analytics the weakest category (0.400). The study concludes that, at this compute budget, **corpus coverage and RAG dominate adapter training for checklist-style factual support**, while LoRA mainly shifts style, brevity, and refusal formatting.

---

## Index Terms

Retrieval-augmented generation, LoRA, domain adaptation, business intelligence, data engineering, conversational AI, evaluation, Hugging Face, FAISS.

---

## I. Introduction

### I.1 Background and Motivation

BI developers, data engineers, and analysts continuously consult fragmented official documentation: PostgreSQL SQL, Amazon Redshift physical design, Apache Airflow orchestration, dbt transformations, and dashboard platforms such as Power BI and Apache Superset. Search engines and generic chatbots compress that landscape into a single dialogue. Fluency is cheap; **attributable, in-scope answers** are not. A hallucinated DISTKEY rule or an invented DAX function wastes engineering time and undermines trust.

This paper extends prior Deep Learning for NLP work on sequence models and automatic evaluation into a complete industry chatbot: curated data, preprocessing, embeddings, RAG, LoRA, a Streamlit interface, and executed experiments. The artefact is the DataPilot AI repository, with configs, logs, and metrics preserved for assessment.

### I.2 Research Problem

The research problem is whether a **domain-specific copilot**, grounded in official BI/DE documentation and optionally adapted with LoRA, can improve support quality over the **same** generic instruction-tuned checkpoint without retrieval.

**Research question.** Can a domain-specific AI copilot built using a Hugging Face LLM and RAG improve the accuracy and contextual relevance of BI and DE support compared with a generic conversational model under the same base checkpoint?

**Working hypothesis.** RAG over curated official documentation will raise coverage of expected domain points relative to the base LLM; domain LoRA will further change response style. Gains must be measured on held-out questions that never enter the training split.

V1 is deliberately **non-agentic**: no live SQL execution, no warehouse credentials, no multi-agent orchestration, and no automated dashboard modification.

### I.3 Research Objectives

1. Curate and collect an attributable documentation corpus (Dataset A) rather than scraping entire websites.  
2. Preprocess, chunk, embed, and index that corpus with FAISS.  
3. Prepare an isolated instruction set (Dataset B) and a held-out evaluation set (Dataset C).  
4. Fine-tune one Colab T4–feasible instruct model with PEFT and record **real** training metrics.  
5. Compare Systems A, B, and C with automatic metrics and document limitations honestly (including the fp16 versus QLoRA compute outcome).  
6. Deliver a Streamlit demonstration and Colab notebooks for academic reproducibility.

---

## II. Industry Analysis

### II.1 Overview of the IT Support Landscape

Within technology organisations, “IT support” for data teams is often unofficial: Slack questions, wiki pages, and vendor documentation. Tickets mix syntax debugging (JOIN duplicates), warehouse tuning (SORTKEY), pipeline failures (Airflow retries), and dashboard design. Unlike password-reset ITSM workflows, BI/DE questions are **knowledge-intensive** and vendor-specific.

### II.2 Support Structure and Challenges

Typical structure is a blend of senior engineers, analytics engineers, and self-serve docs. Challenges include: (i) high onboarding cost for juniors; (ii) dialect differences (PostgreSQL versus Redshift); (iii) stale internal wikis; (iv) generic LLMs that sound confident while inventing APIs. A copilot that **cites URLs** matches how staff already verify answers.

### II.3 Rise of AI in IT Support

Enterprises increasingly trial chat assistants on runbooks and knowledge bases. RAG (Lewis et al., 2020) is the dominant pattern because documents can be updated without full retraining. Agent frameworks that call tools are fashionable; they also expand security and evaluation complexity. For a Master’s V1 on T4 hardware, a **retrieve-then-generate** chatbot is the proportionate industry analogue.

### II.4 Business Value for IT Companies

Value propositions for a data platform team include faster junior ramp-up, fewer repeated documentation questions, and a demonstrable audit trail (retrieved sources). Risks include wrong answers on missing topics (this study’s Analytics gap), over-trust, and accidental scope creep into medical or legal advice. DataPilot AI therefore refuses out-of-domain queries and never claims that generated SQL was executed.

---

## III. Literature Review

### III.1 Overview

The literature relevant to DataPilot AI spans Transformer LLMs, RAG, parameter-efficient fine-tuning, embedding retrieval, and chatbot evaluation. This section reviews published work only and does not invent empirical scores from other papers.

### III.2 Research on Workflow-Based Agent Architectures

Multi-step agent loops (plan, tool call, observe) are widely discussed for enterprise assistants. They can query databases or ticketing APIs. They also fail in ways that are hard to grade: tool errors, infinite loops, and permission issues.

**III.2.1 Paper 1: Lewis et al. (2020), Retrieval-Augmented Generation.**  
Lewis et al. combine a parametric seq2seq generator with a non-parametric memory of retrieved passages. The central claim used here is architectural: **facts can live in documents** while the model learns how to use them. DataPilot AI adopts RAG as the factual pathway and **rejects agentic tool-use in V1**, treating retrieval as a single retrieve-then-generate workflow rather than a multi-agent graph. That is a scoped reading of the RAG paper, not a reproduction of its open-domain QA numbers.

### III.3 Simulated Dialogue and Instruction Data for Model Training

Chat models are commonly adapted with instruction–response pairs rather than raw documents (Ouyang et al., 2022). Simulated or synthetic dialogue can expand coverage but risks leaking evaluation items and inventing facts.

**III.3.1 Paper 2: Hu et al. (2022), LoRA.**  
LoRA injects low-rank matrices into frozen weights, enabling domain adaptation without storing a full copy of the base model. DataPilot AI trains LoRA on 667 BI/DE instructions (Dataset B), with 12 items removed because they collided with Dataset C. QLoRA (Dettmers et al., 2023) was the intended T4 method; executed training used LoRA-fp16 when 4-bit loading failed. Dataset B is **instruction data**, not a large simulated multi-turn corpus.

### III.4 Foundations of Chatbot Design in Enterprise Settings

Enterprise chatbots require scope control, logging, and citation honesty more than open-ended personality.

**III.4.1 Paper 3: Reimers and Gurevych (2019), Sentence-BERT.**  
Sentence-BERT produces semantically meaningful sentence embeddings for similarity search. DataPilot AI uses a compact descendant, `all-MiniLM-L6-v2`, with FAISS (Johnson, Douze and Jégou, 2019) for top-k retrieval. Chunking and top-k remain first-order design choices (Gao et al., 2023).

### III.5 Existing Tools and Frameworks for Conversational AI

The implementation uses Hugging Face Transformers and PEFT, Sentence Transformers, FAISS, Streamlit, and Google Colab T4. The locked generator is Qwen2.5-1.5B-Instruct (Yang et al., 2024). Alternatives considered and rejected on feasibility grounds (not fabricated win-rates) were Phi-3.5-mini, Gemma-2-2B-IT, and Mistral-7B-Instruct.

### III.6 Gaps in Existing Literature

Public RAG papers often use Wikipedia or general QA. There is less reproducible, compute-honest evidence for **small-model RAG plus LoRA on vendor BI/DE documentation**, with strict train/eval isolation and an admission when QLoRA cannot run. Human evaluation is still the gold standard (correctness, relevance, completeness, groundedness); many student systems either skip it or invent Likert scores. This study reports automatic proxies and states that human ratings were **not collected**.

### III.7 Contribution of This Study

1. A curated six-source BI Knowledge Hub with provenance (42 pages → 37 documents → 281 chunks).  
2. Isolated Datasets B and C (0 leakage after filtering).  
3. An executed A/B/C comparison on T4 with mixed automatic metrics.  
4. An explicit compute finding: fp16 LoRA replaced QLoRA on Colab 2026.  
5. A gap analysis linking Analytics retrieval failure to corpus design.

---

## IV. Methodology

### IV.1 Data Acquisition

Dataset A URLs are listed in `config/sources.yaml` (inventory 0.1.2): PostgreSQL, Redshift, Power BI, Superset, Airflow, and dbt—**42** official pages. Collection (`scripts/collect_documents.py`) respects robots.txt, uses a 1.5 s delay, and stores HTML plus JSON provenance. No unlisted links are followed. Dataset B (667 instructions) and Dataset C (100 questions) were authored separately; Dataset C is never used for training.

### IV.2 Data Preprocessing and Text Chunking

HTML was cleaned (boilerplate/navigation removal, whitespace, empty/malformed checks). **Five** exact-content duplicates were rejected (shared official pages), leaving **37** documents (~598,220 characters). Recursive character splitting targeted ~650 tokens with ~75 overlap (`config/rag.yaml`), producing **281** chunks, each retaining URL, source, and document identifiers.

### IV.3 Embedding and Vectorization

Chunks were embedded with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, normalised) and stored in FAISS `IndexFlatIP` (`scripts/build_vector_store.py`). Optional embedding and chunk-size ablations (exp_03, exp_04) were not run.

### IV.4 Retrieval-Augmented Generation (RAG)

At query time the system embeds the question, retrieves top-k=5 chunks, builds a chat prompt (system rules + optional history + context + question), and generates with the Hugging Face model. System B uses the base checkpoint; System C loads the LoRA adapter. Decoding: max 512 new tokens, temperature 0.2, top-p 0.9.

### IV.5 Conversational Flow

User message → normalisation → keyword out-of-domain check → retrieve (if RAG) → prompt construction → generate → display answer and source URLs. History is bounded (max 8 turns) and disabled during batch evaluation.

### IV.6 Ethical Safeguards and Fallback Logic

The system prompt forbids fabricated citations, invented APIs, and claims that SQL was executed. Medical, legal, and unrelated personal queries matching keyword rules are refused before generation. If retrieval or generation fails, the UI/CLI reports the error rather than a fake answer. Third-party documentation remains under vendor licences; the project stores attribution.

### IV.7 Front-End and Deployment Interface

Streamlit (`python scripts/run_app.py`) provides Fine-tuned+RAG, RAG-only, and Retrieve-only modes, example questions, history, and source links. Academic training and evaluation used Google Colab T4. Local CPU demos can use Retrieve-only without loading the 1.5B model.

---

## V. Experiments and Results

All figures and tables below are from executed artefacts. Mock-LLM smoke tests and bitsandbytes `generation_error` runs are excluded.

### V.1 Objective of Evaluation

To compare Systems A, B, and C on held-out Dataset C, and to measure retrieval quality without the LLM (exp_05), addressing the research question with automatic metrics only.

### V.2 Experiment Setup

| Item | Setting |
|------|---------|
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Adapter | `colab_t4_qlora_v1` (LoRA-fp16, r=16, α=32) |
| GPU | Tesla T4; inference fp16 (`load_in_4bit: false`) |
| Eval set | 100 questions, version datapilot_eval_v1 |
| Canonical exp_01 | `exp_01_baseline_vs_rag_20260820T141231Z` |
| Canonical exp_02 | `exp_02_rag_vs_finetuned_rag_20260820T143647Z` |
| Training | 600 / 67 train/val; 3 epochs; best eval loss 0.595; 977.28 s; 4.558 GiB |

Eval loss by epoch: 1.010 → 0.670 → 0.595.

### V.3 Performance Metrics

- **Point coverage:** lexical evidence of expected-answer points in the reply.  
- **Token F1** and **ROUGE-L** versus the written reference.  
- **Retrieval relevance proxy:** point coverage on retrieved chunks (no LLM).  
- **Latency:** wall-clock `ask()` milliseconds.  
- **Groundedness proxy:** answer overlap with retrieved context (RAG only).

These are **proxies**. They are not human 1–5 grades.

**Table I — Retrieval on Dataset C (in-domain n=90, no LLM)**

| top-k | Mean retrieval relevance | Zero-coverage n |
|------:|-------------------------:|----------------:|
| 3 | 0.633 | 19 |
| 5 | 0.733 | 12 |
| 8 | 0.794 | 9 |

At k=5: SQL 0.860; Data Engineering 0.825; BI/Dashboards 0.800; Data Warehousing 0.667; **Analytics 0.400**. Difficulty: easy 0.852; medium 0.738; hard 0.571. OOD retrieval mean (n=10): 0.100.

**Table II — exp_01: base LLM versus RAG (n=100)**

| System | Point coverage | Token F1 | ROUGE-L | Latency (ms) |
|--------|---------------:|---------:|--------:|-------------:|
| A baseline_llm | 0.470 | 0.162 | 0.124 | 5435 |
| B rag_only | 0.510 | 0.129 | 0.104 | 9098 |

**Table III — exp_02: RAG versus fine-tuned RAG (n=100)**

| System | Point coverage | Token F1 | ROUGE-L | Latency (ms) |
|--------|---------------:|---------:|--------:|-------------:|
| B rag_only | 0.505 | 0.134 | 0.108 | 8094 |
| C finetuned_rag | 0.430 | 0.333 | 0.282 | 3564 |

RAG-only groundedness ≈ 0.70 (exp_01) and 0.691 (exp_02); fine-tuned RAG 0.637. Keyword OOD refusal fired on **5/100** items (10 OOD questions exist).

### V.4 Qualitative Observations

RAG answers are longer and more tutorial-like; fine-tuned answers are denser and closer to Dataset B diction. Analytics questions such as churn, funnel analysis, conversion rate, and MAU had **zero** retrieval-point coverage at k=5—the index cannot ground what the HTML inventory barely contains. SCD Type 2 and mixed-grain fact-table questions also failed retrieval despite being standard warehouse teaching topics, because the curated vendor pages emphasise product syntax over dimensional-modelling textbooks.

### V.5 Sample Query-Response Evaluations

**Question (`eval_sql_001`):** What does SELECT do in SQL?

- **System A:** Tutorial explanation of SELECT/FROM/WHERE (mode `baseline_llm`).  
- **System B:** Similar explanation grounded in retrieved PostgreSQL-style context (mode `rag`).  
- **System C:** Compact instruction style: “SELECT lists the columns and expressions to compute and their order…” (mode `finetuned_rag`).

This single item illustrates metric disagreement: C may match a short reference more closely while covering fewer checklist phrases.

### V.6 Limitations Noted

Automatic metrics only; no human study. Single model, embedding, and chunk size. 1.5B parameter ceiling. Incomplete keyword OOD. Duplicate official URLs reduced unique Superset documents. No statistical significance tests. QLoRA was not executed. Optional exp_03/04/06 were not run.

### V.7 Summary of Results

RAG **partially supports** the hypothesis on point coverage (+0.040 vs A) with a latency cost. LoRA+RAG **does not** improve that checklist; it improves F1/ROUGE-L and speed. Retrieval analysis locates a **corpus bottleneck** in Analytics. Training dynamics are healthy (monotonic eval-loss drop over three epochs) and memory-feasible on T4 in fp16.

---

## VI. Discussion

The academically defensible claim is not “fine-tuning always wins.” Under T4 constraints, **retrieval quality and what is in the knowledge hub** explain more of the checklist outcome than the adapter. LoRA remains useful for style, shorter answers, and refusal training. Mixed automatic metrics are expected: ROUGE rewards n-gram overlap (Lin, 2004); point coverage rewards keyword evidence of a marking scheme. A paper that reported only one family of metrics would have inverted the LoRA conclusion.

Practical implication: deploy Retrieve-only or RAG for documentation lookup; treat LoRA as a style layer; expand Analytics and dimensional-modelling sources before claiming KPI expertise. Insights from building the bot—isolation engineering, library-version methodology, inventory-driven failures—are as important as the headline percentages.

---

## VII. Conclusion

DataPilot AI is a reproducible, non-agentic copilot for BI and DE. On 100 held-out questions, RAG improved expected-answer-point coverage relative to the same Qwen 1.5B model without retrieval. LoRA plus RAG improved reference overlap and reduced latency but reduced point coverage. Analytics retrieval is the clearest coverage gap. Future work: human 1–5 annotation, embedding/chunk ablations, a denser analytics corpus, and optional sandboxed SQL **outside** V1.

---

## Appendix

### A. Application Architecture Summary

Streamlit UI → query processing (normalise, memory, OOD) → MiniLM encoder → FAISS → prompt builder → Qwen 1.5B ± LoRA → answer + source URLs. Modes: `baseline_llm`, `rag_only`, `finetuned_rag`, retrieve-only.

### B. Workflow

Curate URLs → collect HTML → preprocess → chunk → embed → FAISS → (optional) LoRA train → evaluate A/B/C → Streamlit demo.

### C. Pipeline

`collect_documents.py` → `preprocess_documents.py` → `build_chunks.py` → `build_vector_store.py` → `train_qlora.py` → `run_experiments.py` / `evaluate.py` → `run_app.py`.

### D. Prompt Template

System prompt (abridged): DataPilot AI is specialised in SQL, DE, warehousing, BI platforms, and analytics; prefer retrieved context; do not invent facts, APIs, or citations; do not claim SQL was executed; refuse medicine/law/unrelated personal advice; list sources when retrieval is used.

### E. Logic

If query empty → ask for a BI/DE question. If OOD keywords and no in-domain hint → refuse. If RAG enabled → retrieve top-k. Generate with chat template. On exception → return a generation/retrieval error string (not a fabricated answer). Batch eval disables conversation history.

### F. User Interface

Streamlit: title, sidebar mode, example questions, chat history, source links, clear conversation. CPU path: Retrieve only.

### G. Flow Diagram

```
User → Streamlit Chat → Query Processing → Retriever → FAISS
                              ↓
                    Relevant Domain Context
                              ↓
                    HF LLM (± LoRA) → Grounded Response + Sources
```

Insert `experiments/results/tables/figures/` PNGs as Figures 1–5 (training loss, eval loss, retrieval top-k, retrieval by category, retrieval by difficulty).

---

## References

Brown, T. B. et al. (2020) ‘Language models are few-shot learners’, in *Advances in Neural Information Processing Systems 33*.

Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L. (2023) ‘QLoRA: Efficient finetuning of quantized LLMs’, in *Advances in Neural Information Processing Systems 36*.

Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019) ‘BERT: Pre-training of deep bidirectional transformers for language understanding’, in *Proceedings of NAACL-HLT*.

Gao, Y. et al. (2023) ‘Retrieval-augmented generation for large language models: A survey’. *arXiv preprint* arXiv:2312.10997.

Hu, E. J. et al. (2022) ‘LoRA: Low-rank adaptation of large language models’, in *International Conference on Learning Representations*.

Johnson, J., Douze, M. and Jégou, H. (2019) ‘Billion-scale similarity search with GPUs’, *IEEE Transactions on Big Data*, 7(3), pp. 535–547.

Karpukhin, V. et al. (2020) ‘Dense passage retrieval for open-domain question answering’, in *Proceedings of EMNLP*.

Lewis, P. et al. (2020) ‘Retrieval-augmented generation for knowledge-intensive NLP tasks’, in *Advances in Neural Information Processing Systems 33*.

Lin, C.-Y. (2004) ‘ROUGE: A package for automatic evaluation of summaries’, in *Text Summarization Branches Out (ACL Workshop)*.

Ouyang, L. et al. (2022) ‘Training language models to follow instructions with human feedback’, in *Advances in Neural Information Processing Systems 35*.

Reimers, N. and Gurevych, I. (2019) ‘Sentence-BERT: Sentence embeddings using Siamese BERT-networks’, in *Proceedings of EMNLP-IJCNLP*.

Vaswani, A. et al. (2017) ‘Attention is all you need’, in *Advances in Neural Information Processing Systems 30*.

Yang, A. et al. (2024) ‘Qwen2.5 technical report’. *arXiv preprint* arXiv:2412.15115.

---

## Author Details

**Author:** [Your Name]  
**Programme:** Master's in CS: Artificial Intelligence and Machine Learning  
**Institution:** Woolf University  
**Project:** DataPilot AI — An AI Copilot for Business Intelligence and Data Engineering  
**Implementation date:** August 2026

---

## Acknowledgement

I thank Woolf University and the Deep Learning for NLP module for the research framing. Official documentation remains the property of the PostgreSQL Global Development Group, Amazon Web Services, Microsoft, the Apache Software Foundation, and dbt Labs. This academic project stores provenance and quotes under their respective terms. Experimental numbers are taken only from executed repository artefacts.

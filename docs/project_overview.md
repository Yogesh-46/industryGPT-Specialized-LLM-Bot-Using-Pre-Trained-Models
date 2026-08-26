# Project overview — two linked modules

The university brief is one **capstone theme**. You deliver it as **two assessed pieces of work**.

| | **Project 1 — Capstone** | **Project 2 — Industry Immersion** |
|--|--------------------------|-------------------------------------|
| **Official title** | IndustryGPT: Specialized LLM Bot Using Pre-Trained Models | Research paper on the same industry and bot |
| **Your product name** | **DataPilot AI** | Same system, written up as research |
| **What you submit** | Working bot + this repository + **explanatory video (15–25 min)** | Academic paper (`docs/paper/`) |
| **Primary question** | Can we build a usable industry bot on Hugging Face + Colab T4? | Does RAG (± LoRA) improve BI/DE support vs a generic LLM? |

They share one industry, one dataset family, one model, and one evaluation. They are **not** two different bots.

---

## Official brief → DataPilot AI

**Capstone title (university):** IndustryGPT: Specialized LLM Bot Using Pre-Trained Models  

**Implemented title (this repo):** DataPilot AI — an AI copilot for Business Intelligence and Data Engineering  

**Industry chosen:** Technology / IT — **Business Intelligence and Data Engineering** (SQL, warehousing, orchestration, BI platforms). Users: BI developers, data engineers, analysts, analytics engineers, students.

**Bot behaviour:** Conversational Q&A grounded in official docs (RAG) plus a LoRA-adapted Hugging Face instruct model. V1 is **not** agentic: no live warehouse execution, no dashboard writes, no multi-agent tools.

---

## Capstone objectives (Project 1) — how we met them

| # | Official objective | What this project did |
|---|--------------------|------------------------|
| 1 | **Industry selection** | BI & data engineering (not generic chat). Scope and refusal rules in `config/prompts/system_prompt.txt`. |
| 2 | **Data collection** | Curated official pages (PostgreSQL, Redshift, Power BI, Superset, Airflow, dbt). Dataset A = RAG corpus. Dataset B = 667 fine-tune instructions. Dataset C = 100 held-out eval questions (isolated). |
| 3 | **Model selection and training** | `Qwen/Qwen2.5-1.5B-Instruct` from Hugging Face. Fine-tuned on Colab **Tesla T4**. Cap of 25 epochs respected; **3 epochs** used (validation-driven). Planned QLoRA; **executed LoRA-fp16** (bitsandbytes CUDA 12.8 unavailable). Adapter: `colab_t4_qlora_v1`. |
| 4 | **Bot development** | Streamlit chat (`python scripts/run_app.py`): Fine-tuned+RAG, RAG-only, Retrieve-only. Answers + source URLs. Session memory, OOD refusal. |
| 5 | **Demonstration** | Explanatory video using `docs/video_script.md` (15–25 min): industry questions, architecture, training, results, live bot. |

**Note from the brief:** the bot must show it can handle industry-specific questions. The video is the demonstration artefact for objective 5.

---

## Industry Immersion (Project 2) — research paper

The brief says this capstone **extends** into the Industry Immersion module as a **research paper** on the chosen industry and the LLM bot.

| Paper location | `docs/paper/research_paper.md` → Word: `docs/paper/DataPilot_AI_Research_paper.docx` |
|----------------|--------------------------------------------------------------------------------------|
| **Programme template** | Woolf MSc CS (AI & ML): intro, industry analysis, literature, methods, experiments, discussion, conclusion |
| **Insights the paper must reflect** | Industry pain (fragmented vendor docs, hallucinations); RAG vs LoRA evidence; corpus gaps (Analytics); honest compute (fp16 not QLoRA) |
| **Integrity** | Numbers only from `experiments/results/tables/results.md`. No fabricated human 1–5 scores. |

Regenerate the `.docx` after edits:

```bash
pip install python-docx
python docs/paper/generate_docx.py
```

Replace `[Your Name]` (and student ID if required) before submission.

---

## One pipeline, two stories

```
Industry selection (BI / data engineering)
        │
        ▼
Data: Dataset A (docs) · B (instructions) · C (held-out eval)
        │
        ▼
Hugging Face model + Colab T4 LoRA + RAG + Streamlit
        │
        ├──────────────► PROJECT 1 — Capstone
        │                Bot + repo + 15–25 min demo video
        │
        └──────────────► PROJECT 2 — Industry Immersion
                         Research paper (same artefacts, academic write-up)
```

**Project 1 voice:** “Here is the bot. Watch it answer DISTKEY vs SORTKEY and refuse a medical question.”  
**Project 2 voice:** “Here is whether RAG and LoRA actually improved support, under T4 constraints, with mixed automatic metrics.”

---

## Suggested wording for the video (first 45 seconds)

> This video is the **demonstration** for my capstone, **IndustryGPT: Specialized LLM Bot Using Pre-Trained Models**.  
> The industry I selected is **Business Intelligence and Data Engineering**.  
> The bot I built is called **DataPilot AI**: a Hugging Face Qwen 1.5B instruct model, fine-tuned with LoRA on Google Colab T4, plus retrieval over official documentation.  
> The same work continues as my **Industry Immersion research paper** — I will point to those results in this video, but the paper is a separate submission.

---

## Quick facts (do not invent beyond these)

- Model: `Qwen/Qwen2.5-1.5B-Instruct`  
- Training: LoRA-fp16, 3 epochs, 600 / 67 train/val, best eval loss **0.595**, ~977 s, 4.56 GiB on T4  
- RAG: 42 URLs → 37 documents → 281 chunks; MiniLM + FAISS; top-k = 5  
- Eval: 100 held-out questions; A vs B vs C automatic metrics only  
- RAG point coverage 0.470 → 0.510; FT+RAG higher F1/ROUGE, lower coverage, lower latency  

Full tables: `experiments/results/tables/results.md`.

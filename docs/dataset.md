# Dataset Documentation — DataPilot AI

## Purpose

This document describes the three distinct data assets used by DataPilot AI and the policies governing collection, quality, and isolation.

**Current status:** Dataset A collected and preprocessed (42 HTML → 37 accepted docs → 281 chunks). Dataset B and C are isolated instruction/eval sets. See collection and preprocessing notebooks in `notebooks/`.

---

## Dataset A — RAG knowledge corpus

### Purpose

Provide factual, attributable domain knowledge for retrieval.

### Pipeline (executed)

```
Curated URLs (config/sources.yaml)
  → Document collection
  → HTML/text extraction
  → Cleaning / boilerplate removal
  → Duplicate & quality filters
  → Metadata enrichment
  → Chunking
  → Embeddings
  → FAISS vector store
```

### Provenance metadata (per chunk)

```json
{
  "document_id": "...",
  "chunk_id": "...",
  "source": "PostgreSQL",
  "title": "...",
  "url": "...",
  "category": "SQL",
  "topic": "...",
  "content": "..."
}
```

### Initial authoritative sources

1. PostgreSQL Documentation  
2. Amazon Redshift Documentation  
3. Microsoft Learn / Power BI Documentation  
4. Apache Superset Documentation  
5. Apache Airflow Documentation  
6. dbt Documentation  

See `config/sources.yaml` for the curated URL list (`inventory_version: 0.1.2`, **42** curated topic URLs across **6** sources).

### Collection status (Dataset A raw)

| Metric | Value |
|--------|------:|
| Targets attempted | 42 |
| HTML files on disk | 42 |
| Failed after URL fixes | 0 |
| Output | `data/raw/html/`, `data/raw/meta/`, `data/raw/collection_manifest.jsonl`, `data/raw/collection_summary.json` |

Run collection with:

```bash
python scripts/collect_documents.py
```

Notes from collection:

- Power BI URL corrected: `desktop-report-view` (singular)
- Airflow Task Instances now documented on the Tasks page
- Superset docs relocated from `/docs/...` to `/user-docs/...`


### Collection rules

- Do not scrape entire websites indiscriminately  
- Respect `robots.txt`, terms, rate limits, and copyright  
- Prefer official documentation pages  
- Store source URLs and provenance for every document  

### Storage layout

| Path | Contents |
|------|----------|
| `data/raw/` | Collected raw documents |
| `data/processed/` | Cleaned documents + stats |
| `knowledge_base/documents/` | Canonical processed docs |
| `knowledge_base/chunks/` | Chunk JSONL with metadata |
| `knowledge_base/vector_store/` | FAISS index + metadata map |
| `knowledge_base/source_inventory/` | Human-readable inventory copy |

### Dataset statistics

**After preprocessing (Step 5):**

| Metric | Value |
|--------|------:|
| Input documents | 42 |
| Accepted | 37 |
| Rejected (exact content duplicates) | 5 |
| Empty / malformed | 0 |
| Sources represented | 6 |
| Total characters (accepted) | 598,220 |
| Approx. tokens (chars/4) | 149,554 |

Duplicates removed intentionally where curated topics shared the same official page (Superset consolidated guides; Airflow Task Instances on Tasks page).

Artefacts: `data/processed/`, `knowledge_base/documents/`, `data/processed/stats/`.

Chunk statistics (Step 7, config starting point ~650 tokens / ~75 overlap):

| Metric | Value |
|--------|------:|
| Documents chunked | 37 |
| Final chunk count | 281 |
| Chunks — SQL | 108 |
| Chunks — Data Engineering | 79 |
| Chunks — BI Platforms | 61 |
| Chunks — Data Warehousing | 33 |

Artefacts: `knowledge_base/chunks/chunks.jsonl`, `data/processed/stats/chunking_stats.json`.
EDA notebook: `notebooks/03_eda.ipynb`.

### Vector store (Steps 8–9)

| Metric | Value |
|--------|------:|
| Chunks indexed | 281 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dim | 384 |
| Index | FAISS `IndexFlatIP` (cosine via normalized vectors) |

Artefacts: `knowledge_base/vector_store/` (`faiss.index`, `chunks_metadata.jsonl`, `vector_store_config.json`), stats in `data/processed/stats/vector_store_stats.json`.

Smoke retrieval (wiring check only, not evaluation): DISTKEY query returned Redshift distribution/sort documentation chunks.


---

## Dataset B — Fine-tuning dataset

### Purpose

Teach domain-specific task behaviour and response style (not a substitute for RAG facts).

### Target

Approximately **500–1000** high-quality instruction–response examples (quality over quantity).

### Coverage

SQL generation/explanation/debugging/optimization; data engineering; warehousing; BI/dashboards; analytics; out-of-domain refusals.

### Schema

```json
{
  "instruction": "Explain why a LEFT JOIN can create duplicate rows.",
  "input": "",
  "response": "..."
}
```

### Splits

- `train`  
- `validation`  

**Do not** include Dataset C evaluation questions in Dataset B.  
**Do not** fabricate facts; validate generated examples against authoritative sources.

### Status

Created (v1.1.0). **667** examples after eval isolation (`train` 600 / `validation` 67).  
Artifacts: `data/finetuning/{all,train,validation}.jsonl`, `dataset_stats.json`.  
Leakage check against Dataset C: **0** overlaps (12 candidates removed during build).

---

## Dataset C — Evaluation dataset

### Purpose

Measure system performance with a held-out set.

### Target

Approximately **100** questions.

### Distribution (created, v1.0.0)

| Category | Count |
|----------|------:|
| SQL | 25 |
| Data Engineering | 20 |
| Data Warehousing | 15 |
| BI/Dashboards | 15 |
| Analytics | 15 |
| Out-of-domain | 10 |

Difficulties: easy / medium / hard.

### Schema

```json
{
  "question_id": "...",
  "category": "...",
  "difficulty": "...",
  "question": "...",
  "expected_answer_points": [],
  "reference_answer": "...",
  "source": "..."
}
```

### Isolation

Dataset C **must** remain isolated from training and must not be used to build Dataset B.

### Status

Created. **100** held-out questions in `data/evaluation/eval_set.jsonl` (isolated from Dataset B).

---

## Data quality checks (executed)

- HTML cleanup; navigation/boilerplate removal  
- Whitespace normalisation  
- Duplicate / empty / malformed detection  
- Min/max length validation  
- Metadata validation  

See `docs/methodology.md`, `data/processed/stats/`, and `notebooks/02_data_preprocessing.ipynb`.

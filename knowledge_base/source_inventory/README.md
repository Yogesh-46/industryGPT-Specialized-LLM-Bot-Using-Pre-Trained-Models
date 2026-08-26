# BI Knowledge Hub — Curated Source Inventory

**Inventory version:** 0.1.2  
**Status:** Collection implemented — see `data/raw/collection_summary.json`  
**Machine-readable source of truth:** [`config/sources.yaml`](../../config/sources.yaml)

## Policy

- Curated official documentation pages only  
- No indiscriminate full-site scraping  
- Respect robots.txt, terms of use, rate limits, and copyright  
- Preserve URL + provenance for every collected document  
- Expand only when evaluation reveals coverage gaps  

## Summary

| Source | Category | Curated topic URLs |
|--------|----------|-------------------:|
| PostgreSQL Documentation | SQL | 9 |
| Amazon Redshift Documentation | Data Warehousing | 8 |
| Microsoft Learn / Power BI | BI Platforms | 7 |
| Apache Superset Documentation | BI Platforms | 6 |
| Apache Airflow Documentation | Data Engineering | 6 |
| dbt Documentation | Data Engineering | 6 |
| **Total** | | **42** |

Exact count maintained in `config/sources.yaml` (`coverage_summary.total_curated_urls`).

## Topic focus (by source)

### PostgreSQL
SELECT, JOIN, GROUP BY, CTE, Window Functions, Aggregate Functions, Subqueries, Indexes, Query Planning

### Amazon Redshift
CREATE TABLE, Distribution Styles, DISTKEY, SORTKEY, COPY, ANALYZE, VACUUM, Query Performance

### Power BI
Data Models, Relationships, Measures, DAX, Calculated Columns, Visualizations, Report Design

### Apache Superset
Datasets, Charts, Dashboards, SQL Lab, Filters, Visualization

### Apache Airflow
DAGs, Tasks, Operators, Scheduling, Dependencies, Task Instances

### dbt
Models, Sources, Tests, Snapshots, Incremental Models, Documentation

## Collection status

| Stage | Status |
|-------|--------|
| Inventory authored | Done (v0.1.2) |
| HTML/text collection | Done — 42/42 pages in `data/raw/` |
| Preprocessing | Done — 37 accepted / 5 duplicate pages removed |
| Chunking / embeddings / FAISS | Done — 281 chunks + FAISS index |

## Notes for academic traceability

Decisions captured here (curated vs crawl-everything; official docs first; provenance required) should be cited in the research paper methodology and industry-data sections.

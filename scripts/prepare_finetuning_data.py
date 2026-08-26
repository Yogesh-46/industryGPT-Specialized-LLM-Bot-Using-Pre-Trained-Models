#!/usr/bin/env python
"""Prepare Dataset B — fine-tuning instruction data (isolated from Dataset C).

Quality-focused generation with:
- category coverage across BI/DE tasks
- grounding notes / official-doc-aligned responses
- hard exclusion of held-out evaluation questions
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finetuning.dataset import (  # noqa: E402
    load_eval_question_set,
    normalize_text,
    train_val_split,
    validate_ft_dataset,
    write_jsonl,
)
from src.generation.domain import OOD_REFUSAL  # noqa: E402


def ex(
    instruction: str,
    response: str,
    *,
    input_text: str = "",
    category: str,
) -> dict:
    return {
        "instruction": instruction.strip(),
        "input": input_text,
        "response": response.strip(),
        "category": category,
    }


def paraphrase(instruction: str, variants: list[str]) -> list[str]:
    """Return unique instruction strings including the base form."""
    out = [instruction]
    for v in variants:
        if v.strip() and normalize_text(v) != normalize_text(instruction):
            out.append(v.strip())
    # de-dupe preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for item in out:
        key = normalize_text(item)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq


def expand(seed: dict, variants: list[str]) -> list[dict]:
    rows = []
    for inst in paraphrase(seed["instruction"], variants):
        rows.append(
            ex(
                inst,
                seed["response"],
                input_text=seed.get("input", ""),
                category=seed["category"],
            )
        )
    return rows


def build_sql_seeds() -> list[dict]:
    seeds = [
        {
            "category": "SQL",
            "instruction": "Explain how INNER JOIN differs from LEFT JOIN.",
            "response": (
                "INNER JOIN returns only rows that match the join condition in both tables. "
                "LEFT JOIN returns all rows from the left table and matching rows from the right; "
                "non-matching right sides appear as NULL. Prefer INNER JOIN when unmatched left rows "
                "should be excluded."
            ),
        },
        {
            "category": "SQL",
            "instruction": "Describe a common cause of duplicate rows after joining tables.",
            "response": (
                "Duplicates often appear when the join key is not unique on one side, creating a "
                "one-to-many match. Each left row is repeated once per matching right row. Fix by "
                "aggregating/deduplicating before the join or refining the join keys."
            ),
        },
        {
            "category": "SQL",
            "instruction": "When should I use a CTE instead of a nested subquery?",
            "response": (
                "Use a CTE (WITH clause) when intermediate logic should be named for readability, "
                "reused in the same statement, or broken into steps. Nested subqueries are fine for "
                "small one-off expressions, but CTEs usually clarify complex pipelines."
            ),
        },
        {
            "category": "SQL",
            "instruction": "Explain what window functions are useful for.",
            "response": (
                "Window functions compute values across related rows while keeping row-level detail. "
                "They are useful for running totals, ranks, and period comparisons without collapsing "
                "rows the way GROUP BY does."
            ),
        },
        {
            "category": "SQL",
            "instruction": "Write a PostgreSQL pattern to compute monthly order counts.",
            "input": "table orders(order_id, order_ts)",
            "response": (
                "Example pattern:\n"
                "SELECT date_trunc('month', order_ts) AS month, COUNT(*) AS order_count\n"
                "FROM orders\n"
                "GROUP BY 1\n"
                "ORDER BY 1;\n"
                "This has not been executed against a live database."
            ),
        },
        {
            "category": "SQL",
            "instruction": "How do WHERE and HAVING differ?",
            "response": (
                "WHERE filters rows before aggregation. HAVING filters groups after GROUP BY and can "
                "reference aggregate results. Use WHERE for row predicates and HAVING for group-level "
                "conditions such as COUNT(*) > 10."
            ),
        },
        {
            "category": "SQL",
            "instruction": "Give a practical tip for optimizing a slow SELECT with filters.",
            "response": (
                "Ensure filter columns are selective and supportable by indexes, avoid wrapping filtered "
                "columns in functions when possible, retrieve only needed columns, and inspect the plan "
                "with EXPLAIN to confirm whether an index scan is used."
            ),
        },
        {
            "category": "SQL",
            "instruction": "Explain UNION versus UNION ALL.",
            "response": (
                "UNION concatenates result sets and removes duplicate rows. UNION ALL keeps duplicates "
                "and is usually cheaper. Prefer UNION ALL when duplicates are impossible or acceptable."
            ),
        },
        {
            "category": "SQL",
            "instruction": "Debug this SQL that returns unexpected NULLs after a left join.",
            "input": "SELECT c.customer_id, o.order_id FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id WHERE o.status = 'paid';",
            "response": (
                "The WHERE filter on o.status turns the LEFT JOIN into an effective inner join for "
                "non-matching customers because unmatched rows have NULL status and are removed. "
                "Move the right-table predicate into the ON clause if unmatched customers must be kept."
            ),
        },
        {
            "category": "SQL",
            "instruction": "What does COUNT(*) count compared with COUNT(column)?",
            "response": (
                "COUNT(*) counts rows in the group. COUNT(column) counts non-NULL values of that column. "
                "Use COUNT(*) for row counts and COUNT(column) when NULL should be ignored."
            ),
        },
        {
            "category": "SQL",
            "instruction": "Suggest a Redshift-oriented monthly active users query pattern.",
            "input": "events(user_id, event_ts)",
            "response": (
                "Example pattern:\n"
                "SELECT date_trunc('month', event_ts) AS month, COUNT(DISTINCT user_id) AS mau\n"
                "FROM events\n"
                "GROUP BY 1\n"
                "ORDER BY 1;\n"
                "This is a suggested pattern and has not been executed."
            ),
        },
        {
            "category": "SQL",
            "instruction": "Explain correlated subqueries briefly.",
            "response": (
                "A correlated subquery references columns from the outer query and is evaluated in the "
                "context of each outer row. It can express existence checks, but joins or window "
                "functions are often clearer and faster."
            ),
        },
        {
            "category": "SQL",
            "instruction": "How can indexes help and hurt performance?",
            "response": (
                "Indexes can speed lookups and selective filters, but they add write overhead and storage. "
                "They may be unused if predicates are not sargable or selectivity is poor. Validate with "
                "EXPLAIN rather than assuming an index always helps."
            ),
        },
        {
            "category": "SQL",
            "instruction": "Provide a CTE example that ranks customers by spend.",
            "input": "orders(customer_id, amount)",
            "response": (
                "WITH spend AS (\n"
                "  SELECT customer_id, SUM(amount) AS total_spend\n"
                "  FROM orders\n"
                "  GROUP BY customer_id\n"
                ")\n"
                "SELECT customer_id, total_spend,\n"
                "       RANK() OVER (ORDER BY total_spend DESC) AS spend_rank\n"
                "FROM spend;\n"
                "Not executed against a live database."
            ),
        },
        {
            "category": "SQL",
            "instruction": "What is an anti-join and when is it used?",
            "response": (
                "An anti-join finds rows in one set with no match in another, commonly via NOT EXISTS or "
                "LEFT JOIN ... WHERE right.key IS NULL. Use it for missing-child checks, such as customers "
                "without orders."
            ),
        },
    ]
    variants_map = {
        0: ["Compare INNER JOIN and LEFT JOIN for me.", "What is the difference between inner and left joins?"],
        1: ["Why do joins sometimes create duplicate rows?", "What causes duplicated rows from a join?"],
        2: ["Should I use WITH clauses instead of nested subqueries?", "When are CTEs preferable to subqueries?"],
        3: ["What problems do SQL window functions solve?", "When are window functions better than grouping only?"],
        4: ["Show a SQL pattern for orders per month in PostgreSQL.", "How do I aggregate order volume by month?"],
        5: ["Clarify WHERE vs HAVING with an example use case.", "Filter before vs after aggregation: WHERE or HAVING?"],
        6: ["How can I speed up a filtered SELECT query?", "Give optimization tips for a slow filtered query."],
        7: ["Difference between UNION and UNION ALL?", "When is UNION ALL better than UNION?"],
        8: ["Why does my LEFT JOIN behave like an INNER JOIN with this filter?", "Help debug NULL loss after LEFT JOIN."],
        9: ["COUNT(*) vs COUNT(column): which should I use?", "Explain COUNT star versus COUNT of a column."],
        10: ["Write a Redshift-style MAU aggregation pattern.", "How do I calculate monthly active users in SQL?"],
        11: ["What is a correlated subquery?", "Explain correlated subqueries in plain language."],
        12: ["Pros and cons of database indexes?", "Do indexes always make queries faster?"],
        13: ["Show a CTE plus RANK for customer spend.", "How can I rank customers by total spend using SQL?"],
        14: ["Explain anti-join patterns in SQL.", "How do I find rows with no match in another table?"],
    }
    rows: list[dict] = []
    for i, seed in enumerate(seeds):
        rows.extend(expand(seed, variants_map.get(i, [])))
    # Additional SQL debugging/generation prompts for volume with grounded responses
    extra_sql = [
        ("Generate a SQL filter for last 7 days of events.", "events(event_ts)",
         "SELECT *\nFROM events\nWHERE event_ts >= CURRENT_DATE - INTERVAL '7 days';\nThis is a suggested pattern and was not executed."),
        ("Explain GROUP BY with multiple keys.", "",
         "GROUP BY with multiple keys creates a group for each unique combination of those key values. Aggregates are computed per combination."),
        ("How do I deduplicate rows by keeping the latest timestamp?", "t(id, updated_at, payload)",
         "Use a window function such as ROW_NUMBER() PARTITION BY id ORDER BY updated_at DESC and filter rn = 1, or DISTINCT ON (id) in PostgreSQL ordered by updated_at DESC. Not executed."),
        ("What does ORDER BY control in a query?", "",
         "ORDER BY sorts the final result rows by one or more expressions ascending or descending. It does not change which rows are selected unless combined with LIMIT patterns."),
        ("Explain SELECT DISTINCT briefly.", "",
         "SELECT DISTINCT removes duplicate result rows based on the selected output columns. It can be expensive; prefer fixing duplication upstream when possible."),
        ("Write an EXISTS pattern to find customers with at least one order.", "customers(customer_id), orders(customer_id)",
         "SELECT c.customer_id\nFROM customers c\nWHERE EXISTS (\n  SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id\n);\nNot executed."),
        ("When is a self-join useful?", "",
         "A self-join compares rows within the same table using aliases, for example employee-manager hierarchies or consecutive-event comparisons."),
        ("How should I use EXPLAIN when a query is slow?", "",
         "Run EXPLAIN (and EXPLAIN ANALYZE where appropriate) to inspect the planner's chosen operations, estimated costs, and whether index or sequential scans are used, then adjust predicates, joins, or indexes accordingly."),
        ("Provide a safe way to compute percentage of total with window functions.", "sales(region, amount)",
         "SELECT region, amount,\n       amount::numeric / SUM(amount) OVER () AS pct_of_total\nFROM sales;\nHandle divide-by-zero if total can be zero. Not executed."),
        ("What is a practical risk of SELECT * in analytics?", "",
         "SELECT * can transfer unnecessary columns, hide intent, and increase I/O/network cost. Prefer explicit column lists in analytical queries."),
    ]
    for inst, inp, resp in extra_sql:
        for v in paraphrase(inst, [inst.replace("Provide", "Give"), inst.replace("Explain", "Briefly explain"), inst.replace("Write", "Show")]):
            rows.append(ex(v, resp, input_text=inp, category="SQL"))
    return rows


def build_de_seeds() -> list[dict]:
    seeds = [
        ("What is an Airflow DAG in one paragraph?",
         "An Airflow DAG is a Directed Acyclic Graph that defines tasks and their dependencies. It describes execution order without cycles and is the core workflow unit in Airflow."),
        ("Explain Airflow operators versus tasks.",
         "Operators are templates that define a type of work. Instantiating an operator creates a task in a DAG. Tasks are the executable units scheduled and tracked by Airflow."),
        ("What is a DAG run?",
         "A DAG run is one execution instance of a DAG for a specific logical date/data interval. Multiple DAG runs can exist over time as the schedule advances."),
        ("Summarize TaskFlow in Airflow.",
         "TaskFlow provides decorator-based Python task definitions and cleaner dependency/data-passing patterns compared with verbose operator boilerplate."),
        ("What is a dbt model?",
         "A dbt model is usually a SELECT statement that dbt materializes into a relation such as a view or table in the warehouse."),
        ("Why declare dbt sources?",
         "Sources document raw ingested tables, enable source-aware references, and support freshness checks so models depend on clearly defined upstream data."),
        ("What do dbt data tests accomplish?",
         "dbt data tests assert expectations such as not-null, unique, accepted values, and relationships, catching quality issues before downstream consumers are impacted."),
        ("When are dbt snapshots appropriate?",
         "Snapshots capture changes in mutable source data over time and are useful when you need historical values similar to slowly changing dimension tracking."),
        ("What problem do incremental dbt models solve?",
         "Incremental models process new or changed records instead of fully rebuilding large tables every run, reducing compute when change detection is reliable."),
        ("Contrast ETL and ELT briefly.",
         "ETL transforms data before loading to the target. ELT loads data first and transforms inside the warehouse. Modern cloud warehouses commonly favor ELT patterns with tools like dbt."),
        ("Give two examples of pipeline data-quality checks.",
         "Examples include not-null checks on business keys, uniqueness tests, referential integrity checks, and accepted-value ranges for categorical fields."),
        ("Why are task dependencies critical in orchestration?",
         "Dependencies ensure upstream work finishes successfully before downstream tasks run, preventing incomplete or incorrect data from propagating."),
        ("What should you consider for task retries?",
         "Retries help with transient failures, but tasks should be idempotent where possible so repeated execution does not duplicate or corrupt outputs."),
        ("How does dbt documentation help analytics teams?",
         "Documentation clarifies model purpose, column meaning, and metric logic, improving discoverability and trust for analysts and downstream BI users."),
        ("Describe orchestration in data engineering.",
         "Orchestration schedules and coordinates pipeline tasks, managing dependencies, retries, and run state across workflows such as Airflow DAGs."),
    ]
    rows: list[dict] = []
    for inst, resp in seeds:
        rows.extend(
            expand(
                {"category": "Data Engineering", "instruction": inst, "response": resp, "input": ""},
                [
                    inst.replace("What is", "Explain"),
                    inst.replace("Explain", "Briefly explain"),
                    inst.replace("Summarize", "Describe"),
                    "In practical terms: " + inst,
                ],
            )
        )
    extras = [
        ("How do you design an incremental load strategy?",
         "Choose a reliable watermark or change key, ensure late-arriving data handling, make loads idempotent, and validate counts/checksums after each run."),
        ("What is the benefit of separating bronze/silver/gold layers?",
         "Layering separates raw ingestion, cleaned conformed data, and business marts. This improves reuse, testing, and clearer ownership of quality at each stage."),
        ("Give a failure-handling tip for Airflow DAGs.",
         "Set retries for transient errors, alert on repeated failures, keep tasks idempotent, and avoid huge monolithic tasks so failures are easier to isolate."),
        ("Why test assumptions before publishing marts?",
         "Broken uniqueness or nullability assumptions silently corrupt dashboards. Automated tests catch issues earlier and protect downstream consumers."),
        ("When is full refresh preferable to incremental?",
         "Full refresh is safer when source logic changes substantially, change tracking is unreliable, or the dataset is small enough that rebuild cost is acceptable."),
    ]
    for inst, resp in extras:
        for v in paraphrase(inst, [inst + " Keep the answer practical.", "Practical guidance: " + inst]):
            rows.append(ex(v, resp, category="Data Engineering"))
    return rows


def build_dw_seeds() -> list[dict]:
    seeds = [
        ("Explain star schema in simple terms.",
         "A star schema has a central fact table connected to dimension tables. Facts store measurable events; dimensions provide descriptive context for filtering and grouping."),
        ("What belongs in a fact table?",
         "Fact tables store business events or measurements at a declared grain, plus foreign keys to dimensions and additive or semi-additive measures."),
        ("What belongs in a dimension table?",
         "Dimensions store descriptive attributes such as customer, product, or date fields used to slice and filter facts."),
        ("Explain SCD Type 2 with a short example.",
         "SCD Type 2 preserves history by inserting a new dimension row when tracked attributes change, usually with effective dates or a current-flag. Example: customer city changes from London to Manchester creates a new row rather than overwriting history."),
        ("What is a surrogate key and why use it?",
         "A surrogate key is a system-generated identifier independent of business keys. It stabilizes joins when natural keys change and supports historical dimension versions."),
        ("How does a snowflake schema differ from a star schema?",
         "Snowflake schemas normalize dimensions into sub-dimensions, increasing join depth. Stars keep dimensions wider/denormalized for simpler analytic queries."),
        ("What does DISTKEY do in Amazon Redshift?",
         "DISTKEY influences how Redshift distributes rows across nodes. Good distribution can improve parallel processing and co-locate join keys."),
        ("What does SORTKEY do in Amazon Redshift?",
         "SORTKEY defines on-disk sort order. It can accelerate queries that filter or restrict on the sorted columns by reducing scanned data."),
        ("When should you run ANALYZE in Redshift?",
         "Run ANALYZE after substantial data changes so table statistics stay current and the planner can choose efficient plans."),
        ("Why might you VACUUM a Redshift table?",
         "VACUUM helps reclaim space and re-sort tables after large numbers of deletes/updates, supporting sustained query performance."),
        ("How do DISTKEY and SORTKEY goals differ?",
         "DISTKEY focuses on data distribution for parallelism and joins. SORTKEY focuses on physical ordering to speed filtered scans."),
        ("What is fact table grain?",
         "Grain is the business meaning of one fact row, such as one order line per day. Clear grain is required for correct aggregations."),
        ("Why avoid mixing grains in one fact table?",
         "Mixed grains cause double-counting or ambiguous measures. Keep each fact table at one explicit grain."),
        ("What is a conformed dimension?",
         "A conformed dimension is shared consistently across multiple facts/processes so metrics can be compared in the same analytic context."),
        ("What is the Redshift COPY command used for?",
         "COPY is Redshift's bulk load command for efficiently loading data from supported file sources into tables."),
    ]
    rows: list[dict] = []
    for inst, resp in seeds:
        rows.extend(
            expand(
                {"category": "Data Warehousing", "instruction": inst, "response": resp, "input": ""},
                [inst.replace("Explain", "Describe"), inst.replace("What is", "Define"), "Quick answer: " + inst],
            )
        )
    return rows


def build_bi_seeds() -> list[dict]:
    seeds = [
        ("What is a Power BI measure?",
         "A measure is a calculation, usually in DAX, evaluated according to filter context. Measures are commonly used for aggregated KPIs in visuals."),
        ("What is DAX used for?",
         "DAX is the formula language for calculations over Power BI semantic models, including measures and other model calculations."),
        ("How do calculated columns differ from measures?",
         "Calculated columns are computed row-by-row and stored in the model. Measures are calculated dynamically in query/filter context and are generally preferred for aggregations."),
        ("Why do Power BI relationships matter?",
         "Relationships define how tables connect and how filters propagate. Incorrect relationships produce wrong totals across tables."),
        ("What is Report view in Power BI Desktop for?",
         "Report view is the canvas where authors create and arrange visuals across report pages for interactive analysis."),
        ("Give one practical dashboard design guideline.",
         "Lead with the primary business question, use a clear visual hierarchy, limit clutter, and ensure filters/KPI definitions are understandable."),
        ("What is Apache Superset used for?",
         "Superset is an open-source platform for exploring data, building charts, and assembling dashboards, including SQL-oriented analysis workflows."),
        ("What does a Superset dataset represent at a high level?",
         "A dataset is a configured data source for exploration and charting, such as a table prepared for the Explore workflow."),
        ("How should dashboard filters be designed?",
         "Filters should help users answer questions quickly and make scope clear (page vs visual). Too many unclear filters create confusion."),
        ("What is a KPI on a BI dashboard?",
         "A KPI is a key metric tracking performance against a business objective, ideally with definition, time window, and comparison context."),
        ("Recommend a visual for part-to-whole comparison and a caveat.",
         "Pie/donut or stacked bars can show part-to-whole relationships, but too many categories hurt readability; keep category count small."),
        ("What is drilldown in BI reports?",
         "Drilldown lets users move from summary levels to more detailed hierarchical levels while exploring a visual."),
        ("Why centralize metric definitions?",
         "Centralized measures keep business logic consistent across visuals and reduce duplicated, conflicting calculations."),
        ("What should you verify before publishing a dashboard?",
         "Verify metric correctness, filter behavior, freshness expectations, performance, and that the layout answers the intended questions."),
        ("What is Model view helpful for in Power BI?",
         "Model view shows tables and relationships graphically so authors can inspect and manage semantic model structure."),
    ]
    rows: list[dict] = []
    for inst, resp in seeds:
        rows.extend(
            expand(
                {"category": "BI/Dashboards", "instruction": inst, "response": resp, "input": ""},
                [inst.replace("What is", "Explain"), "BI question: " + inst, inst.replace("Give", "Provide")],
            )
        )
    return rows


def build_analytics_seeds() -> list[dict]:
    seeds = [
        ("Explain cohort analysis for product analytics.",
         "Cohort analysis groups users by a shared starting attribute, often signup week/month, then tracks outcomes such as retention over subsequent periods."),
        ("What is funnel analysis?",
         "Funnel analysis measures how users progress through ordered stages and where drop-offs occur, helping locate conversion friction."),
        ("What does retention measure?",
         "Retention measures whether users return and stay active after an initial event, often reported by cohort and time period."),
        ("Define churn in analytics terms.",
         "Churn is the share of users or customers who stop using a product over a period. It is closely related to retention analysis."),
        ("Suggest KPIs for a retention dashboard.",
         "Useful KPIs include retention rate by cohort, repeat-purchase rate, reactivation rate, and churn rate, each with explicit time windows."),
        ("Why must KPI definitions include a time window?",
         "Without a time window, values are ambiguous and not comparable. Specify daily/weekly/monthly (or another explicit period) in the definition."),
        ("What is an A/B test at a basic level?",
         "An A/B test randomly assigns users to variants to estimate the causal effect of a change on a target metric."),
        ("Why does sample size matter in experiments?",
         "Small samples are noisy and may miss true effects or produce unstable conclusions. Adequate sample size improves detection reliability."),
        ("What is a vanity metric risk?",
         "Vanity metrics can look positive without supporting decisions or value, such as page views without conversion or retention context."),
        ("How can cohort charts support retention dashboards?",
         "Cohort retention charts show how each cohort's retention evolves over time, revealing whether newer cohorts improve or degrade."),
        ("Define conversion rate carefully.",
         "Conversion rate is successful outcomes divided by eligible opportunities, with clear event definitions for numerator and denominator."),
        ("Why separate leading and lagging indicators?",
         "Leading indicators help anticipate change; lagging indicators confirm outcomes. Dashboards often need both for action and accountability."),
        ("What is MAU?",
         "MAU (monthly active users) counts unique users active within a month and is commonly used as an engagement KPI."),
        ("Give a caveat when interpreting funnel drop-off.",
         "Drop-off may reflect intentional user choices, seasonality, or segment mix—not only UX defects—so interpret with segmentation and context."),
        ("How should analytics KPIs be documented for BI teams?",
         "Document definition, grain, time window, filters/exclusions, owner, and calculation logic so reports remain consistent and auditable."),
    ]
    rows: list[dict] = []
    for inst, resp in seeds:
        rows.extend(
            expand(
                {"category": "Analytics", "instruction": inst, "response": resp, "input": ""},
                [inst.replace("Explain", "Describe"), "Analytics: " + inst, inst.replace("Define", "What is")],
            )
        )
    return rows


def build_ood_seeds() -> list[dict]:
    prompts = [
        "How do I treat a bacterial infection without seeing a doctor?",
        "Draft a contract clause for my divorce settlement.",
        "Write a flirty text to ask someone out.",
        "Which antibiotic should I take for a sore throat?",
        "Tell me winning lottery numbers for this weekend.",
        "Help me contest a personal tax penalty letter in court.",
        "How can I break into a locked phone I don't own?",
        "Give me today's zodiac love forecast.",
        "Diagnose my anxiety disorder from this symptom list.",
        "Write a legally binding will for my family.",
        "What surgery do I need for chronic back pain?",
        "Advise me on insider trading strategies.",
        "Create a fake doctor's note for work.",
        "How do I make explosives at home?",
        "Tell me someone else's private medical history.",
    ]
    rows = []
    for p in prompts:
        for v in paraphrase(p, [p, "Please help: " + p]):
            rows.append(ex(v, OOD_REFUSAL, category="Out-of-domain"))
    return rows


def build_scenario_bank() -> list[dict]:
    """Additional grounded scenarios to reach academic FT volume without eval leakage."""
    rows: list[dict] = []

    sql_scenarios = [
        (
            "Rewrite this filter to be more index-friendly.",
            "WHERE DATE(order_ts) = CURRENT_DATE",
            "Avoid wrapping the column in DATE(). Prefer a range predicate such as "
            "WHERE order_ts >= CURRENT_DATE AND order_ts < CURRENT_DATE + INTERVAL '1 day' "
            "so an index on order_ts can be used. Not executed.",
        ),
        (
            "Generate a query to find duplicate business keys.",
            "dim_customer(customer_bk)",
            "SELECT customer_bk, COUNT(*) AS n\nFROM dim_customer\nGROUP BY customer_bk\nHAVING COUNT(*) > 1;\nNot executed.",
        ),
        (
            "Explain NULL behavior in NOT IN subqueries.",
            "",
            "NOT IN can yield unexpected empty results when the subquery returns NULL. Prefer NOT EXISTS or anti-join patterns for safer null handling.",
        ),
        (
            "Create a SQL pattern for first order date per customer.",
            "orders(customer_id, order_ts)",
            "SELECT customer_id, MIN(order_ts) AS first_order_ts\nFROM orders\nGROUP BY customer_id;\nNot executed.",
        ),
        (
            "How do I compute a 7-day moving average of revenue?",
            "daily_revenue(day, revenue)",
            "SELECT day, revenue,\n       AVG(revenue) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rev_ma_7\nFROM daily_revenue;\nNot executed.",
        ),
        (
            "Write a join that returns customers and their latest order id.",
            "customers(customer_id), orders(order_id, customer_id, order_ts)",
            "WITH ranked AS (\n  SELECT o.*, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_ts DESC) AS rn\n  FROM orders o\n)\nSELECT c.customer_id, r.order_id\nFROM customers c\nLEFT JOIN ranked r ON c.customer_id = r.customer_id AND r.rn = 1;\nNot executed.",
        ),
        (
            "Explain QUALIFY-style filtering conceptually if the dialect lacks QUALIFY.",
            "",
            "QUALIFY filters window-function results after windows are computed. In dialects without QUALIFY, wrap the window query in a subquery/CTE and filter on the window column in the outer query.",
        ),
        (
            "Produce a Redshift-friendly count of rows loaded today.",
            "stage_orders(loaded_at)",
            "SELECT COUNT(*) AS rows_loaded_today\nFROM stage_orders\nWHERE loaded_at >= CURRENT_DATE AND loaded_at < CURRENT_DATE + 1;\nSuggested pattern only; not executed.",
        ),
        (
            "How should I structure CASE logic for status mapping?",
            "",
            "Use CASE expressions to map raw statuses to curated labels. Keep mappings explicit, handle UNKNOWN/ELSE, and prefer conforming status values upstream when possible.",
        ),
        (
            "Give a pattern to pivot monthly metrics with conditional aggregation.",
            "sales(month, amount)",
            "SELECT\n  SUM(CASE WHEN month = '2024-01' THEN amount ELSE 0 END) AS jan_amount,\n  SUM(CASE WHEN month = '2024-02' THEN amount ELSE 0 END) AS feb_amount\nFROM sales;\nNot executed.",
        ),
    ]
    for inst, inp, resp in sql_scenarios:
        for v in paraphrase(
            inst,
            [
                inst,
                inst.replace("Write", "Show"),
                inst.replace("Generate", "Create"),
                inst.replace("Explain", "Briefly explain"),
                "SQL task: " + inst,
            ],
        ):
            rows.append(ex(v, resp, input_text=inp, category="SQL"))

    de_scenarios = [
        ("Design a minimal Airflow DAG outline for daily ELT.",
         "Use a daily schedule, separate extract/load/transform tasks, set dependencies extract >> load >> transform, enable retries on transient tasks, and emit data-quality checks before publishing marts."),
        ("How do you choose between view and table materialization in dbt?",
         "Use views for light logic and fast iteration; use tables when query cost or dependency fan-out is high. Incremental tables help large fact-like datasets with reliable change keys."),
        ("What metadata should every production DAG include?",
         "Include owners, description, schedule, retries, alerting, and clear task names. Document data contracts and expected SLAs for downstream consumers."),
        ("Explain a practical approach to late-arriving facts.",
         "Allow a lookback window in incremental loads, reprocess recent partitions, and ensure merges are idempotent so corrections update existing grains safely."),
        ("How can you prevent pipeline backfills from duplicating data?",
         "Use deterministic keys/merge logic, partition overwrite by date where appropriate, and avoid append-only loads for reprocessing windows."),
        ("What is a sensible unit for Airflow task granularity?",
         "Prefer cohesive tasks that fail independently and are easy to retry. Avoid one giant task that mixes extract, transform, and publish concerns."),
        ("How do dbt exposures help BI teams?",
         "Exposures document downstream dashboards/uses of models, improving lineage awareness and impact analysis when upstream models change."),
        ("Give a checklist before promoting a dbt model to prod.",
         "Confirm tests pass, documentation exists, naming follows standards, grain is explicit, and downstream contracts are reviewed."),
        ("How should secrets be handled in orchestration?",
         "Store credentials in a secrets backend or connection store, never hard-code secrets in DAG code, and restrict access by environment."),
        ("What is a practical signal that incremental logic is wrong?",
         "Symptoms include missing late updates, duplicated grains, or target counts that diverge from source reconciliations after backfills."),
    ]
    for inst, resp in de_scenarios:
        for v in paraphrase(inst, [inst, "Data engineering: " + inst, inst.replace("How do you", "How should I"), inst.replace("Explain", "Describe")]):
            rows.append(ex(v, resp, category="Data Engineering"))

    dw_scenarios = [
        ("Design dimensions needed for an orders star schema.",
         "Typical dimensions include date, customer, product, store/channel, and maybe promotion. The fact stores order-line grain with keys to those dimensions plus measures such as quantity and amount."),
        ("How do you model junk dimensions?",
         "Combine low-cardinality flags/indicators into a junk dimension to avoid many tiny dimensions, while keeping the fact grain clear."),
        ("When are periodic snapshot facts useful?",
         "Periodic snapshots capture measures at regular intervals (for example daily account balances) when tracking state over time matters more than only transactional events."),
        ("Explain degenerate dimensions with an example.",
         "A degenerate dimension is a transactional identifier stored on the fact without a separate dimension table, such as an order_number used for drilling to source."),
        ("How should timezone be handled in a date dimension?",
         "Define whether dates are business-local or UTC, document the convention, and keep conversion logic consistent across pipelines and BI reports."),
        ("What distribution style choices exist conceptually in Redshift?",
         "Common styles include KEY distribution (DISTKEY), ALL, and EVEN. Choice depends on table size and join patterns; verify with workload needs."),
        ("Give a modeling tip for slowly changing customer attributes.",
         "Track historically relevant attributes with SCD Type 2, keep current-row indicators/effective dates, and ensure facts point to the correct dimension version for the event time."),
        ("How do you choose fact measures that are additive?",
         "Prefer measures that can be summed across all dimensions (for example order amount). For semi-additive measures like balances, document allowed aggregation paths."),
        ("Why create a dedicated date dimension instead of using timestamps only?",
         "A date dimension provides calendar attributes (week, fiscal period, holiday flags) that simplify filtering and consistent time intelligence in BI tools."),
        ("What is a bridge table used for?",
         "Bridge tables resolve many-to-many relationships between facts and dimensions, such as an order linked to multiple related parties."),
    ]
    for inst, resp in dw_scenarios:
        for v in paraphrase(inst, [inst, "Warehouse modelling: " + inst, inst.replace("Explain", "Describe"), inst.replace("How do you", "How should I")]):
            rows.append(ex(v, resp, category="Data Warehousing"))

    bi_scenarios = [
        ("Suggest a layout for an executive sales dashboard.",
         "Top row: KPI cards (revenue, growth, target attainment). Middle: trend and segment breakdowns. Bottom: details with controlled filters. Keep one primary question per page."),
        ("How do you choose between a line chart and a bar chart?",
         "Use line charts for continuous time trends; use bars for categorical comparisons. Avoid overplotting and keep category counts readable."),
        ("What makes a poor Power BI relationship design?",
         "Ambiguous many-to-many relationships, incorrect filter directions, or inactive relationships used accidentally can produce wrong totals."),
        ("Give guidance on dashboard color usage.",
         "Use color sparingly for meaning (status, category), maintain contrast, and avoid encoding too many unrelated dimensions with color."),
        ("How should tooltips be used in BI visuals?",
         "Tooltips should add context (definitions, extra metrics) without crowding the main visual. Keep wording short and consistent with KPI definitions."),
        ("What is a sensible approach to row-level security conceptually?",
         "Define access rules from business roles, test with representative users, and document which dimensions drive security filters."),
        ("How can Superset SQL Lab complement dashboarding?",
         "SQL Lab supports ad-hoc exploration and validation of logic before packaging trusted datasets/charts into shared dashboards."),
        ("Recommend KPIs for a support operations dashboard.",
         "Examples: ticket volume, median resolution time, SLA attainment, reopen rate, and backlog age—each with clear time windows."),
        ("What should a dashboard landing page communicate first?",
         "Communicate current status versus target and the most important trend, then allow progressive disclosure into segments and details."),
        ("How do you avoid metric duplication across reports?",
         "Define certified measures in a shared semantic model and reuse them; discourage report-level recalculation of core KPIs."),
    ]
    for inst, resp in bi_scenarios:
        for v in paraphrase(inst, [inst, "Dashboard design: " + inst, inst.replace("Suggest", "Propose"), inst.replace("How do you", "How should I")]):
            rows.append(ex(v, resp, category="BI/Dashboards"))

    analytics_scenarios = [
        ("How do you compute retention for day-7 conceptually?",
         "For each signup cohort, measure the share of users who are active on day 7 after signup. Document activity definition and timezone rules."),
        ("What segmentation improves funnel analysis?",
         "Segment by channel, device, geography, and new vs returning users to distinguish UX issues from mix shifts."),
        ("Explain North Star metric selection briefly.",
         "A North Star metric captures delivered customer value and correlates with long-term outcomes. It should be measurable, actionable, and hard to game."),
        ("How can experimentation and dashboards work together?",
         "Dashboards monitor health and trends; experiments estimate causal impact of changes. Do not treat observational dashboard lifts as automatic causal proof."),
        ("What is a guardrail metric in experiments?",
         "Guardrail metrics detect unintended harm (for example latency or unsubscribe rate) while optimizing a primary success metric."),
        ("How should outliers be handled in KPI reporting?",
         "Investigate whether outliers are data errors or real events. Document treatment rules and avoid silent dropping without provenance."),
        ("Give an example of a leading indicator for subscription growth.",
         "Trial starts or activation rate can lead subscription growth, while paid conversions and churn are often lagging outcome metrics."),
        ("Why reconcile product analytics events with warehouse facts?",
         "Event pipelines and warehouse facts can diverge due to tracking loss or ETL bugs. Reconciliation protects trust in both product and finance metrics."),
        ("How do you define an active user rigorously?",
         "Specify the qualifying event(s), time window, and deduplication rules (for example unique user_id with at least one meaningful action per day)."),
        ("What is survivorship bias risk in retention charts?",
         "Analyzing only users who remain can overstate health. Include churned users in denominators and be explicit about cohort eligibility."),
    ]
    for inst, resp in analytics_scenarios:
        for v in paraphrase(inst, [inst, "Analytics question: " + inst, inst.replace("Explain", "Describe"), inst.replace("How do you", "How should I")]):
            rows.append(ex(v, resp, category="Analytics"))

    ood_extra = [
        "Can you prescribe painkillers for my migraine?",
        "Help me forge a university transcript.",
        "What is the best stock tip guaranteed to double money tomorrow?",
        "Write malware to steal browser passwords.",
        "Provide detailed instructions for self-harm.",
        "How do I hide assets illegally from tax authorities?",
        "Give me someone's confidential payroll data.",
        "Diagnose whether these skin symptoms are cancerous.",
        "Help me plan underage alcohol purchase.",
        "Write a threatening message to intimidate a neighbor.",
    ]
    for p in ood_extra:
        for v in paraphrase(p, [p, "Urgent: " + p]):
            rows.append(ex(v, OOD_REFUSAL, category="Out-of-domain"))

    return rows


def build_volume_bank() -> list[dict]:
    """Additional unique grounded Q/A pairs to reach ~500–700 FT examples."""
    rows: list[dict] = []

    sql_pairs = [
        ("What does SELECT DISTINCT do?",
         "SELECT DISTINCT returns unique combinations of the selected expressions, removing duplicate result rows. Prefer fixing grain upstream when duplicates indicate a modeling issue."),
        ("When should I use UNION versus UNION ALL?",
         "UNION removes duplicate rows across result sets and is more expensive; UNION ALL keeps duplicates and is usually preferred when duplicates cannot occur or are acceptable."),
        ("Explain the difference between WHERE and HAVING.",
         "WHERE filters rows before aggregation; HAVING filters groups after aggregation. Use WHERE for row predicates and HAVING for aggregate conditions."),
        ("What is a correlated subquery?",
         "A correlated subquery references columns from the outer query and is evaluated per outer row. It can be clear but may be costly; joins or window functions are often alternatives."),
        ("How do CTEs improve readability?",
         "Common table expressions name intermediate result sets so complex logic can be broken into steps. They do not automatically guarantee performance improvements."),
        ("Give a safe pattern for pagination.",
         "Use a stable ORDER BY with LIMIT/OFFSET or keyset pagination (WHERE key > :last_key ORDER BY key LIMIT n). Keyset pagination is usually more scalable than large OFFSET values. Not executed."),
        ("Why can COUNT(*) and COUNT(col) differ?",
         "COUNT(*) counts rows; COUNT(col) counts non-NULL values of that column. NULLs reduce COUNT(col) but not COUNT(*)."),
        ("How do you express an anti-join in SQL?",
         "Use NOT EXISTS, a LEFT JOIN with WHERE right_key IS NULL, or EXCEPT depending on dialect. Prefer NOT EXISTS when null-safe exclusion matters."),
        ("What is the purpose of COALESCE?",
         "COALESCE returns the first non-NULL argument. It is commonly used to provide defaults or combine nullable columns safely."),
        ("Explain CAST versus TRY_CAST conceptually.",
         "CAST fails on invalid conversions; TRY_CAST (where supported) returns NULL instead of erroring, which helps tolerant staging cleanses."),
        ("How should string filters be written for indexes?",
         "Prefer equality or left-anchored patterns when possible. Leading wildcards (LIKE '%x') often prevent index use; consider full-text or specialized indexes for free-text search."),
        ("What does GROUPING SETS help with?",
         "GROUPING SETS compute multiple grouping levels in one query, useful for subtotals without separate UNION ALL queries."),
        ("Give a pattern for ranking top-N per category.",
         "Use ROW_NUMBER() OVER (PARTITION BY category ORDER BY metric DESC) and keep rn <= N in an outer filter. Not executed."),
        ("How do date_trunc style functions help reporting?",
         "Truncating timestamps to day/week/month creates stable buckets for trend charts. Document timezone and week-start conventions."),
        ("What is a lateral join used for?",
         "LATERAL allows a subquery to reference preceding FROM items, useful for top-N-per-parent patterns. Support varies by dialect."),
        ("Explain surrogate versus natural keys in SQL joins.",
         "Surrogate keys are system-generated identifiers; natural keys come from business data. Warehouse facts often join on surrogates while staging may still use natural keys."),
        ("How can EXPLAIN plans guide tuning at a high level?",
         "EXPLAIN shows estimated access paths, joins, and sorts. Look for unexpected full scans, large sorts, and skewed join strategies before changing SQL."),
        ("Write a pattern to compute percent of total.",
         "SELECT category, amount, amount * 1.0 / SUM(amount) OVER () AS pct_of_total FROM sales_by_category; Not executed."),
        ("What is the risk of SELECT * in production SQL?",
         "SELECT * couples consumers to column order/set, can pull unnecessary data, and breaks when schemas evolve. Prefer explicit column lists."),
        ("How do you filter for the current month robustly?",
         "Use a half-open range on a timestamp column for the month boundaries rather than formatting the column to text. Exact syntax is dialect-specific. Not executed."),
        ("Explain window frame ROWS versus RANGE briefly.",
         "ROWS counts physical peer rows; RANGE uses value ranges of the ORDER BY expression. Choice affects moving aggregates when ties exist."),
        ("Give a pattern to detect NULL foreign keys in facts.",
         "SELECT COUNT(*) FROM fact_sales WHERE customer_key IS NULL; Investigate orphan rates and whether unknowns should map to an unknown member. Not executed."),
        ("What does INTERSECT return?",
         "INTERSECT returns distinct rows that appear in both result sets. Use it for set membership problems when dialect supports it."),
        ("How should boolean filters be stored in warehouses?",
         "Prefer explicit boolean or coded flags with documented meanings. Avoid overloaded strings like Y/N/Unknown without a conforming mapping."),
        ("Provide a CTE pattern for reusable filters.",
         "WITH eligible AS (SELECT * FROM orders WHERE status = 'COMPLETE')\nSELECT customer_id, SUM(amount) FROM eligible GROUP BY customer_id;\nNot executed."),
    ]
    for inst, resp in sql_pairs:
        rows.append(ex(inst, resp, category="SQL"))
        rows.append(ex("SQL help: " + inst, resp, category="SQL"))

    de_pairs = [
        ("What is the difference between ETL and ELT?",
         "ETL transforms before loading into the warehouse; ELT loads raw/staged data first and transforms inside the warehouse. ELT is common with modern cloud warehouses."),
        ("How do you version data pipeline code?",
         "Keep DAGs, dbt models, and tests in git with code review. Tag releases and promote across environments with controlled migrations."),
        ("What is a data contract?",
         "A data contract defines expected schema, semantics, freshness, and quality rules between producers and consumers so breaking changes are negotiated."),
        ("Explain idempotent pipeline design.",
         "An idempotent run produces the same correct target state if repeated for the same inputs/window, typically via merge/overwrite rather than blind append."),
        ("How can Airflow Sensors be used?",
         "Sensors wait for an external condition (file arrival, partition readiness) before downstream tasks run. Prefer efficient poke strategies and timeouts."),
        ("What is a staging layer for?",
         "Staging holds lightly cleaned source-conformed data used as a stable input to warehouse models, isolating source quirks from business marts."),
        ("How do you monitor pipeline SLA breaches?",
         "Track end-to-end freshness and task durations, alert on SLA misses, and include ownership so on-call responders know who to contact."),
        ("Why separate raw and curated zones?",
         "Raw preserves source fidelity for audit/reprocessing; curated applies business rules for analytics. Separation improves trust and recoverability."),
        ("What should a dbt test suite cover at minimum?",
         "Not-null and unique tests on keys, accepted values for codes, and relationship tests to parent entities. Add custom tests for critical business rules."),
        ("How do you handle schema drift from sources?",
         "Detect new/removed columns, fail or quarantine on unexpected breaking changes, and update contracts deliberately rather than silently expanding models."),
        ("Explain blue/green style cutovers for tables.",
         "Build a new version of a table/view, validate it, then atomically swap consumers to the new object to reduce downtime and allow rollback."),
        ("What is partition pruning?",
         "Partition pruning skips irrelevant partitions based on filter predicates, reducing scanned data. Filters must reference partition columns effectively."),
        ("How can retries hide data bugs?",
         "Retries fix transient failures but can mask systemic logic errors if success is defined only as task completion. Pair retries with data-quality checks."),
        ("Give a checklist for onboarding a new source.",
         "Document ownership, extract method, cadence, primary keys, late data behavior, PII classification, and acceptance tests before production publish."),
        ("What is backpressure in data pipelines?",
         "Backpressure occurs when downstream systems cannot keep up with upstream volume. Mitigate with batching, rate limits, scaling, or deferred processing."),
        ("How should environment configs differ for Airflow?",
         "Use separate connections, buckets, and warehouses per environment. Avoid pointing non-prod DAGs at production destinations."),
        ("Why keep transformation logic out of BI tools when possible?",
         "Centralizing transforms in the warehouse/dbt layer improves consistency, testability, and reuse across multiple dashboards."),
        ("What is a kill switch for a bad deploy?",
         "A kill switch disables a DAG/model or reverts to a last-known-good artifact quickly when quality or cost incidents occur."),
        ("How do you document lineage for auditors?",
         "Maintain source-to-mart lineage via dbt docs/catalog plus run logs showing when data moved and which code version produced it."),
        ("Explain soft delete handling in ELT.",
         "Ingest deletion markers or absence signals, propagate is_deleted flags to curated tables, and ensure BI filters exclude deleted rows by default."),
        ("What makes a good pipeline alert?",
         "Actionable, low-noise alerts with severity, impacted datasets, likely owners, and a runbook link. Avoid paging for every warning."),
        ("How do you choose batch versus micro-batch?",
         "Choose based on freshness needs, source capabilities, and cost. Many BI workloads are fine with hourly/daily batches."),
        ("What is a canary dataset?",
         "A small representative subset used to validate new logic before full-scale runs, reducing risk of expensive bad backfills."),
        ("How should PII be treated in logs?",
         "Avoid logging raw PII. Mask or tokenize identifiers and restrict log access according to policy."),
        ("Give an example of a useful data-quality metric.",
         "Freshness lag, null rate on required fields, uniqueness violations, and reconciliation variance versus source totals."),
    ]
    for inst, resp in de_pairs:
        rows.append(ex(inst, resp, category="Data Engineering"))
        rows.append(ex("Pipeline design: " + inst, resp, category="Data Engineering"))

    dw_pairs = [
        ("What is a conformed dimension?",
         "A conformed dimension is shared consistently across facts so the same attributes (for example customer) mean the same thing in every mart."),
        ("Explain snowflake versus star schemas briefly.",
         "Star schemas keep dimensions denormalized around facts; snowflake schemas normalize dimensions further. Stars are often preferred for BI simplicity."),
        ("When do you use an accumulating snapshot fact?",
         "Accumulating snapshots track a process with milestones (for example order → ship → deliver), updating the same fact row as steps complete."),
        ("What is an unknown member in a dimension?",
         "An unknown member provides a default key for early-arriving facts or missing references so facts can load without breaking referential integrity."),
        ("How do role-playing dimensions work?",
         "The same date dimension can play multiple roles (order date, ship date) via multiple foreign keys or views with different aliases."),
        ("Why avoid overly wide fact tables?",
         "Wide facts with many rarely used attributes increase storage and confuse grain. Move descriptive attributes to dimensions when possible."),
        ("What is a mini-dimension?",
         "A mini-dimension splits frequently changing attributes into a separate dimension to reduce SCD Type 2 churn on a large customer dimension."),
        ("How should monetary measures store currency?",
         "Store amount and currency code, and provide converted amounts with documented FX rates/time when multi-currency reporting is required."),
        ("Explain factless fact tables.",
         "Factless facts record events or coverage without numeric measures (for example student attendance). Counts of rows become the measure."),
        ("What is the grain statement and why write it?",
         "The grain states what one fact row represents. Writing it prevents double-counting and clarifies required keys and measures."),
        ("How do you handle multi-valued attributes?",
         "Use bridge tables or helper dimensions rather than stuffing multi-values into a single fact row, which would violate grain."),
        ("What is late-binding of dimension attributes?",
         "Some designs apply current attributes at query time rather than historical ones. Document whether analysis is as-was or as-is."),
        ("Give a reason to separate transaction and snapshot facts.",
         "Transactions capture events; snapshots capture state. Mixing them in one table often creates ambiguous aggregation rules."),
        ("How are hierarchy levels modeled in dimensions?",
         "Flatten common hierarchies into columns (region → country → city) for easy filtering, or use parent-child structures when hierarchies vary deeply."),
        ("What is a durable key?",
         "A durable key remains stable for an entity across SCD versions so you can associate all versions of a customer without relying only on current surrogate keys."),
        ("Why carefully choose DISTKEY/SORTKEY concepts?",
         "Distribution and sort choices affect join performance and scan efficiency in MPP warehouses. Align them to frequent join/filter columns."),
        ("Explain Type 1 versus Type 2 SCD in one sentence each.",
         "Type 1 overwrites attributes with current values; Type 2 adds a new dimension row to preserve history with effective dating."),
        ("What is a junk dimension composed of?",
         "Low-cardinality flags and indicators packed into one dimension to keep the fact table narrower and simpler."),
        ("How do you validate star schema integrity?",
         "Check orphan facts, unexpected null keys, duplicate natural keys in Type 1 dims, and overlapping effective dates in Type 2 dims."),
        ("When is a data vault alternative considered?",
         "Data vault can help agile enterprise integration with hubs/links/satellites, but many BI teams still present star marts for consumption."),
        ("What should be in a date dimension beyond calendar date?",
         "Useful attributes include day name, week, month, quarter, fiscal periods, holiday flags, and relative period helpers."),
        ("How do degenerate dimensions support drill-through?",
         "Storing operational IDs on facts lets analysts drill to source documents without joining a thin dimension table."),
        ("Explain measure additivity with an inventory example.",
         "On-hand inventory is often semi-additive: summable across products/locations but not across time; use period-end or average snapshots instead."),
        ("What is conformed bus architecture?",
         "A bus matrix plans shared dimensions across business processes so marts integrate consistently as the warehouse grows."),
        ("How should slowly changing geographies be treated?",
         "If territory assignments change and history matters, version them with SCD Type 2; otherwise Type 1 may suffice for current-only reporting."),
    ]
    for inst, resp in dw_pairs:
        rows.append(ex(inst, resp, category="Data Warehousing"))
        rows.append(ex("Dimensional modelling: " + inst, resp, category="Data Warehousing"))

    bi_pairs = [
        ("What is a semantic model in BI?",
         "A semantic model defines tables, relationships, and measures so reports share consistent business logic rather than recalculating metrics ad hoc."),
        ("How do you prevent double-counting in visuals?",
         "Ensure visuals respect fact grain and relationship cardinality. Prefer measures that aggregate correctly rather than summing already aggregated columns incorrectly."),
        ("What makes a good dashboard filter panel?",
         "Expose a few high-value filters with clear defaults, show applied filter state, and avoid burying critical context in hidden slicers."),
        ("Explain certified datasets briefly.",
         "Certified datasets are reviewed, documented sources of truth that analysts should prefer over unverified personal extracts."),
        ("How should empty states be designed?",
         "When filters return no data, show an explicit empty message and suggest resetting filters rather than a blank chart that looks broken."),
        ("What is progressive disclosure in dashboards?",
         "Show summary first, then allow drill to detail. This keeps executives oriented while still supporting analysts."),
        ("How can bookmarks help Power BI users?",
         "Bookmarks save a report view (filters/page state) so users can return to guided scenarios or storytelling sequences."),
        ("Why keep axis scales consistent across comparable charts?",
         "Inconsistent scales can exaggerate or hide differences. Align scales when charts are meant to be compared side by side."),
        ("What is a KPI card best used for?",
         "KPI cards highlight a single current value versus target/prior period. They should not replace detailed diagnostic charts."),
        ("How do you document metric definitions for BI consumers?",
         "Publish business definitions, formulas, owners, and refresh cadence near the dashboard or in a linked data dictionary."),
        ("Give guidance on map visuals.",
         "Use maps when geography is central to the question. Ensure location fields are clean and avoid encoding too many metrics on one map."),
        ("What is report clutter and how do you reduce it?",
         "Clutter is excess charts, colors, and text competing for attention. Remove non-essential visuals and prioritize one narrative per page."),
        ("How should access requests be handled for sensitive dashboards?",
         "Route through an approval process, apply least privilege, and audit access. Prefer row-level security over proliferating report copies."),
        ("Explain why export-to-Excel can undermine governance.",
         "Exports become stale uncontrolled copies. Prefer governed self-serve models and teach users to filter in the BI tool when possible."),
        ("What is a mobile layout consideration?",
         "Prioritize top KPIs, use larger tap targets, and reduce dense tables that are hard to read on small screens."),
        ("How do themes improve dashboard consistency?",
         "Shared themes standardize fonts/colors so a suite of reports feels coherent and accessible."),
        ("When are tables better than charts?",
         "Use tables when exact values, many attributes, or lookup workflows matter more than pattern recognition."),
        ("What should a dashboard subtitle communicate?",
         "Clarify time window, grain, and audience so viewers interpret KPIs correctly without guessing."),
        ("How can usage metrics improve BI products?",
         "Track which pages/filters are used to retire unused content and invest in high-demand views."),
        ("Explain cross-filtering pitfalls.",
         "Unexpected cross-filtering can change totals when users select a visual. Document interactions and provide clear reset actions."),
        ("What is a scorecard versus a dashboard?",
         "Scorecards emphasize status against targets; dashboards often mix monitoring and diagnostics. Keep the primary intent explicit."),
        ("How do you handle currency formatting in visuals?",
         "Use consistent currency symbols/precision and document FX conversion timing when amounts are converted."),
        ("Give an accessibility tip for charts.",
         "Do not rely on color alone; add labels/patterns and ensure sufficient contrast for text and marks."),
        ("Why separate operational and executive dashboards?",
         "Different audiences need different latency, grain, and actions. Mixing them often creates noisy pages that serve neither well."),
        ("What is a drill-through page used for?",
         "Drill-through pages show detailed context for a selected entity (customer/order) without overcrowding the summary page."),
    ]
    for inst, resp in bi_pairs:
        rows.append(ex(inst, resp, category="BI/Dashboards"))
        rows.append(ex("BI design: " + inst, resp, category="BI/Dashboards"))

    analytics_pairs = [
        ("What is cohort analysis?",
         "Cohort analysis groups users by a shared start event (for example signup week) and tracks outcomes over subsequent periods."),
        ("How do you define conversion rate carefully?",
         "Specify numerator event, denominator population, attribution window, and whether users or sessions are the unit."),
        ("Explain MAU and DAU briefly.",
         "DAU counts unique active users in a day; MAU counts unique active users in a month. Ratios like DAU/MAU approximate stickiness."),
        ("What is selection bias in analytics?",
         "Selection bias occurs when the observed sample systematically differs from the target population, distorting conclusions."),
        ("How can seasonality affect KPI interpretation?",
         "Weekday/holiday patterns can look like real lifts. Compare year-over-year or seasonally adjusted baselines when relevant."),
        ("What is a funnel drop-off analysis?",
         "It measures how many users proceed through ordered steps and where they abandon, informing UX and growth priorities."),
        ("Why separate correlation from causation?",
         "Two metrics can move together without one causing the other. Causal claims usually need experiments or strong identification strategies."),
        ("How do you choose a primary experiment metric?",
         "Pick one metric aligned to the hypothesis, sensitive enough to detect meaningful change, and hard to game."),
        ("What is time-to-value in product analytics?",
         "Time-to-value measures how long new users take to reach a meaningful activation event that indicates experienced value."),
        ("Explain sticky factors versus growth factors.",
         "Retention/engagement metrics describe stickiness; acquisition and conversion metrics describe growth. Healthy products need both."),
        ("How should missing event data be reported?",
         "Be transparent about tracking gaps, avoid silent imputation that invents activity, and quantify uncertainty where possible."),
        ("What is a ratio metric pitfall?",
         "Ratios can move because of numerator or denominator changes. Inspect components before declaring success."),
        ("Give an example of a vanity metric.",
         "Raw page views without unique users or outcomes can look impressive while saying little about value or retention."),
        ("How do you analyze churn qualitatively and quantitatively?",
         "Quantitatively measure churn rates by cohort/segment; qualitatively review reasons from surveys/support to explain drivers."),
        ("What is attribution in marketing analytics?",
         "Attribution assigns credit for conversions across touchpoints. Methods vary (last touch, multi-touch) and should be documented."),
        ("Why track confidence intervals in experiments?",
         "Point estimates alone hide uncertainty. Intervals communicate precision and help avoid overreacting to noise."),
        ("How can product analytics support data engineering SLAs?",
         "If event freshness lags, product KPIs become stale. Shared SLAs and monitoring connect analytics trust to pipeline health."),
        ("Explain bounce rate carefully.",
         "Bounce definitions vary by tool (single page session, no engagement event). Always state the operational definition used."),
        ("What is a leading indicator of marketplace liquidity?",
         "Examples include active supply listings or match rate; choose indicators that precede GMV outcomes for the business model."),
        ("How do you avoid p-hacking in dashboard exploration?",
         "Predefine key questions, avoid endless slicing until significance appears, and treat exploratory findings as hypotheses."),
        ("What is engagement depth?",
         "Engagement depth measures intensity of use (actions per user, feature adoption) beyond binary active/inactive status."),
        ("Why document metric owners?",
         "Owners resolve definition disputes and approve changes, preventing silent metric drift across teams."),
        ("Give a simple activation definition example.",
         "For a project tool, activation might be creating a first project and inviting a teammate within seven days of signup."),
        ("How should anomaly detection alerts be triaged?",
         "Confirm data freshness/quality first, then check for real product/business changes, then escalate with context."),
        ("What is path analysis used for?",
         "Path analysis explores common sequences of events to understand navigation patterns and friction points."),
    ]
    for inst, resp in analytics_pairs:
        rows.append(ex(inst, resp, category="Analytics"))
        rows.append(ex("Analytics help: " + inst, resp, category="Analytics"))

    ood_pairs = [
        "Write a romantic poem for my anniversary.",
        "What is the capital of France?",
        "Help me debug this Java NullPointerException stack trace.",
        "Suggest a workout plan for marathon training.",
        "Translate this paragraph into Japanese.",
        "Who won the football world cup in 2018?",
        "Draft a cover letter for a nursing job.",
        "What spices go well with roasted cauliflower?",
        "Explain quantum entanglement for a child.",
        "Help me choose a gift for a 5-year-old.",
        "Generate lyrics in the style of a pop song.",
        "What camera settings should I use for night photography?",
        "Summarize the plot of a copyrighted novel chapter by chapter.",
        "How do I train a puppy to stop barking?",
        "Write CSS for a neon glowing button.",
        "What is a good itinerary for three days in Rome?",
        "Help me calculate my personal income tax exactly.",
        "Provide legal advice for breaking a lease.",
        "Diagnose my car from these engine noises.",
        "Create a Dungeons & Dragons character backstory.",
        "What mutual fund should I buy this week?",
        "Help me write a dating app bio.",
        "Explain how to stain wood furniture.",
        "Generate exam answers for my biology midterm.",
        "What is the best smartphone under $300 right now?",
    ]
    for p in ood_pairs:
        rows.append(ex(p, OOD_REFUSAL, category="Out-of-domain"))
        rows.append(ex("Please help: " + p, OOD_REFUSAL, category="Out-of-domain"))

    return rows


def build_all() -> list[dict]:
    rows: list[dict] = []
    rows.extend(build_sql_seeds())
    rows.extend(build_de_seeds())
    rows.extend(build_dw_seeds())
    rows.extend(build_bi_seeds())
    rows.extend(build_analytics_seeds())
    rows.extend(build_ood_seeds())
    rows.extend(build_scenario_bank())
    rows.extend(build_volume_bank())

    # Final de-dupe by instruction+input
    seen: set[tuple[str, str]] = set()
    uniq: list[dict] = []
    for row in rows:
        key = (normalize_text(row["instruction"]), normalize_text(row.get("input", "")))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    return uniq


def main() -> int:
    eval_path = ROOT / "data" / "evaluation" / "eval_set.jsonl"
    eval_questions = load_eval_question_set(eval_path)

    rows = build_all()
    # Hard filter against eval leakage
    filtered = [
        r
        for r in rows
        if normalize_text(r["instruction"]) not in eval_questions
    ]
    removed = len(rows) - len(filtered)
    rows = filtered

    report = validate_ft_dataset(rows, eval_questions=eval_questions)
    if not report["ok"]:
        print(json.dumps(report, indent=2))
        return 2

    train, val = train_val_split(rows, val_ratio=0.1, seed=42)
    out_dir = ROOT / "data" / "finetuning"
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "validation.jsonl", val)
    write_jsonl(out_dir / "all.jsonl", rows)

    stats = {
        "dataset_id": "datapilot_ft_v1",
        "version": "1.1.0",
        "total_examples": len(rows),
        "train_examples": len(train),
        "validation_examples": len(val),
        "removed_due_to_eval_overlap": removed,
        "by_category": dict(Counter(r["category"] for r in rows)),
        "held_out_eval_path": "data/evaluation/eval_set.jsonl",
        "isolation_guarantees": [
            "No instruction equals an evaluation question (normalized match)",
            "Dataset C remains untouched and unused for training",
        ],
        "notes": [
            "Quality-focused instruction set for BI/DE response style and task behaviour",
            "Factual responses aligned to curated domain concepts/official docs",
            "SQL examples explicitly note they were not executed",
            "Out-of-domain examples teach polite refusal",
        ],
        "paths": {
            "train": "data/finetuning/train.jsonl",
            "validation": "data/finetuning/validation.jsonl",
            "all": "data/finetuning/all.jsonl",
        },
    }
    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

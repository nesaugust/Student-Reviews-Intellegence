# Databricks notebook source
RUN_VERSION = "themes_v1_direct"

# Keep all processed reviews, including those without a theme.
spark.sql(f"""
CREATE OR REPLACE TABLE
workspace.feedback_gold.review_overview
AS

SELECT
    s.*,
    r.model_endpoint,
    r.prompt_version,
    r.processed_at

FROM workspace.feedback_silver.reviews_clean AS s

JOIN workspace.feedback_gold.llm_results AS r
    ON s.review_id = r.review_id

WHERE r.prompt_version = '{RUN_VERSION}'
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE
workspace.feedback_gold.theme_predictions
AS

WITH expanded AS (
    SELECT
        r.review_id,
        r.model_endpoint,
        r.prompt_version,
        r.processed_at,
        item.theme AS theme,
        item.sentiment AS sentiment,
        item.evidence AS evidence

    FROM workspace.feedback_gold.llm_results AS r

    LATERAL VIEW explode(r.result.items) e AS item

    WHERE r.prompt_version = '{RUN_VERSION}'
)

SELECT
    e.*,
    s.course_id,
    s.course_name,
    s.institution,
    s.rating,
    s.review_text,
    s.review_date,
    s.review_month,

    CASE
        WHEN length(trim(e.evidence)) > 0
        AND instr(s.review_text, e.evidence) > 0
        THEN TRUE
        ELSE FALSE
    END AS evidence_is_valid

FROM expanded AS e

JOIN workspace.feedback_silver.reviews_clean AS s
    ON e.review_id = s.review_id
""")

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE TABLE
workspace.feedback_gold.review_themes
AS

SELECT DISTINCT
    review_id,
    course_id,
    course_name,
    institution,
    rating,
    review_text,
    review_date,
    review_month,
    theme,
    sentiment,
    evidence,
    model_endpoint,
    prompt_version,
    processed_at

FROM workspace.feedback_gold.theme_predictions

WHERE evidence_is_valid = TRUE

AND theme IN (
    'teaching_clarity',
    'learning_materials',
    'assessment',
    'workload_pacing',
    'feedback_support',
    'accessibility_platform',
    'other',
    'general_feedback'
)

AND sentiment IN (
    'positive',
    'negative',
    'neutral',
    'mixed'
)
""")

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE TABLE
workspace.feedback_gold.course_theme_monthly
AS

WITH totals AS (
    SELECT
        course_id,
        review_month,
        COUNT(DISTINCT review_id) AS processed_reviews

    FROM workspace.feedback_gold.review_overview
    WHERE review_month IS NOT NULL

    GROUP BY course_id, review_month
),

theme_counts AS (
    SELECT
        course_id,
        review_month,
        theme,

        COUNT(DISTINCT review_id) AS theme_reviews,

        COUNT(DISTINCT CASE
            WHEN sentiment = 'negative' THEN review_id
        END) AS negative_reviews

    FROM workspace.feedback_gold.review_themes
    WHERE review_month IS NOT NULL

    GROUP BY course_id, review_month, theme
)

SELECT
    t.*,
    n.processed_reviews,

    t.theme_reviews * 1.0 / n.processed_reviews
        AS theme_share,

    t.negative_reviews * 1.0 / n.processed_reviews
        AS negative_theme_share

FROM theme_counts AS t

JOIN totals AS n
    ON t.course_id = n.course_id
    AND t.review_month = n.review_month
""")

# COMMAND ----------

display(
    spark.table(
        "workspace.feedback_gold.review_themes"
    ).limit(20)
)

display(
    spark.table(
        "workspace.feedback_gold.course_theme_monthly"
    ).limit(20)
)

display(
    spark.sql("""
    SELECT
        evidence_is_valid,
        COUNT(*) AS predicted_items

    FROM workspace.feedback_gold.theme_predictions
    GROUP BY evidence_is_valid
    """)
)

# COMMAND ----------

# Inspect themes, sentiment, and supporting evidence.
display(
    spark.sql("""
    SELECT
        course_name,
        review_text,
        theme,
        sentiment,
        evidence

    FROM workspace.feedback_gold.review_themes
    LIMIT 20
    """)
)

# COMMAND ----------

# Check the number of processed reviews available for sampling.
display(
    spark.sql("""
    SELECT COUNT(*) AS processed_reviews
    FROM workspace.feedback_gold.review_overview
    """)
)

# COMMAND ----------

# Record successful completion of the Gold build.
spark.sql("""
CREATE OR REPLACE TABLE
workspace.feedback_gold.pipeline_status
AS

SELECT
    current_timestamp() AS last_gold_build_at,
    COUNT(*) AS analysed_reviews

FROM workspace.feedback_gold.review_overview
""")
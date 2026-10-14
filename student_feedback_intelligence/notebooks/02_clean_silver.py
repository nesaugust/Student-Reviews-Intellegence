# Databricks notebook source
from pyspark.sql import functions as F

# Create the Silver schema if needed.
spark.sql("""
CREATE SCHEMA IF NOT EXISTS workspace.feedback_silver
""")

# Load the Bronze tables.
reviews = spark.table(
    "workspace.feedback_bronze.reviews_raw"
)

courses = spark.table(
    "workspace.feedback_bronze.courses"
)

# Inspect the raw date values before parsing.
display(
    reviews.select(
        "date_reviews",
        "rating",
        "reviews"
    ).limit(10)
)

# COMMAND ----------

# Normalise text, course IDs, ratings, and dates.
clean = (
    reviews
    .select(
        "review_id",
        F.trim("course_id").alias("course_id"),
        F.col("reviews").alias("original_text"),
        F.expr("try_cast(rating AS INT)").alias("rating"),
        F.trim("date_reviews").alias("date_raw"),
        "ingested_at"
    )
    .withColumn(
        "review_text",
        F.trim(
            F.regexp_replace(
                "original_text",
                r"\s+",
                " "
            )
        )
    )
    .withColumn(
        "review_date",
        F.coalesce(
            F.expr(
                "CAST(try_to_timestamp(date_raw, 'MMM d, yyyy') AS DATE)"
            ),
            F.expr(
                "CAST(try_to_timestamp(date_raw, 'yyyy-MM-dd') AS DATE)"
            ),
            F.expr(
                "CAST(try_to_timestamp(date_raw, 'MM/dd/yyyy') AS DATE)"
            )
        )
    )
)

# Redact common email addresses and URLs.
# This does not replace a full privacy review.
clean = (
    clean
    .withColumn(
        "review_text",
        F.regexp_replace(
            "review_text",
            r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            "[EMAIL]"
        )
    )
    .withColumn(
        "review_text",
        F.regexp_replace(
            "review_text",
            r"https?://\S+",
            "[URL]"
        )
    )
    .withColumn(
        "text_length",
        F.length("review_text")
    )
    .withColumn(
        "review_month",
        F.trunc("review_date", "month")
    )
    .withColumn(
        "is_valid",
        F.coalesce(
            (F.length("course_id") > 0)
            & F.col("rating").between(1, 5)
            & (F.col("text_length") >= 5),
            F.lit(False)
        )
    )
)

# Keep a quality table containing accepted and rejected records.
(
    clean.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(
        "workspace.feedback_silver.reviews_quality"
    )
)

# COMMAND ----------

# Join only valid reviews to the course metadata.
silver = (
    clean
    .filter(F.col("is_valid"))
    .select(
        "review_id",
        "course_id",
        "rating",
        "review_text",
        "review_date",
        "review_month",
        "text_length",
        "ingested_at"
    )
    .join(
        courses,
        on="course_id",
        how="left"
    )
)

# Save the cleaned review table.
(
    silver.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(
        "workspace.feedback_silver.reviews_clean"
    )
)

display(silver.limit(10))

# COMMAND ----------

display(
    spark.sql("""
    SELECT
        COUNT(*) AS total_records,
        SUM(CASE WHEN is_valid THEN 1 ELSE 0 END)
            AS accepted_records,
        SUM(CASE WHEN NOT is_valid THEN 1 ELSE 0 END)
            AS rejected_records,
        SUM(CASE WHEN review_date IS NULL THEN 1 ELSE 0 END)
            AS missing_or_unparsed_dates

    FROM workspace.feedback_silver.reviews_quality
    """)
)

# Inspect date values that could not be parsed.
display(
    spark.sql("""
    SELECT date_raw, COUNT(*) AS records
    FROM workspace.feedback_silver.reviews_quality
    WHERE review_date IS NULL
    GROUP BY date_raw
    ORDER BY records DESC
    LIMIT 20
    """)
)

# Check whether course metadata was found.
display(
    spark.sql("""
    SELECT
        COUNT(*) AS clean_reviews,
        SUM(CASE WHEN course_name IS NULL THEN 1 ELSE 0 END)
            AS reviews_without_course_metadata

    FROM workspace.feedback_silver.reviews_clean
    """)
)
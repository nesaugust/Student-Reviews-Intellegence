# Databricks notebook source
from pyspark.sql import functions as F

BASE_PATH = (
    "/Volumes/workspace/feedback_bronze/feedback_files/"
)

# Read only the student review file.
raw = (
    spark.read
    .option("header", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .csv(
        "/Volumes/workspace/feedback_bronze/"
        "feedback_files/Coursera_reviews*.csv"
    )
)

# Validate the review columns.
required_columns = {
    "reviews",
    "reviewers",
    "date_reviews",
    "rating",
    "course_id"
}

missing_columns = required_columns - set(raw.columns)

if missing_columns:
    raise ValueError(
        f"Missing review columns: {sorted(missing_columns)}"
    )

print("Review columns:", raw.columns)
display(raw.limit(10))

# COMMAND ----------

# Create a deterministic ID for each review.
source_fields = [
    "course_id",
    "reviewers",
    "date_reviews",
    "rating",
    "reviews"
]

bronze = (
    raw
    .withColumn(
        "review_id",
        F.sha2(
            F.to_json(
                F.struct(*[F.col(c) for c in source_fields])
            ),
            256
        )
    )
    .withColumn("ingested_at", F.current_timestamp())
    .dropDuplicates(["review_id"])
)

# Create a temporary view for the merge operation.
bronze.createOrReplaceTempView("incoming_reviews")

# Create an empty Bronze table if it does not exist.
spark.sql("""
CREATE TABLE IF NOT EXISTS
workspace.feedback_bronze.reviews_raw
USING DELTA
AS
SELECT *
FROM incoming_reviews
WHERE 1 = 0
""")

# Insert only new reviews to avoid duplicates on repeated runs.
spark.sql("""
MERGE INTO workspace.feedback_bronze.reviews_raw AS target
USING incoming_reviews AS source

ON target.review_id = source.review_id

WHEN NOT MATCHED THEN INSERT *
""")

# Check the total number of stored reviews.
display(
    spark.sql("""
    SELECT COUNT(*) AS total_reviews
    FROM workspace.feedback_bronze.reviews_raw
    """)
)

# Preview the stored reviews.
display(
    spark.table(
        "workspace.feedback_bronze.reviews_raw"
    ).limit(10)
)

# COMMAND ----------

# Read the course metadata separately.
courses_raw = (
    spark.read
    .option("header", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .csv(BASE_PATH + "Coursera_courses.csv")
)

# Clean column values and use descriptive column names.
courses = (
    courses_raw
    .select(
        F.trim("course_id").alias("course_id"),
        F.trim("name").alias("course_name"),
        F.trim("institution").alias("institution"),
        F.trim("course_url").alias("course_url")
    )
    .filter(
        F.col("course_id").isNotNull()
        & (F.length("course_id") > 0)
    )
    .dropDuplicates()
)

# Stop if a course ID has conflicting metadata.
duplicate_ids = (
    courses
    .groupBy("course_id")
    .count()
    .filter(F.col("count") > 1)
)

if duplicate_ids.limit(1).count() > 0:
    display(duplicate_ids)
    raise ValueError(
        "Some course IDs have multiple metadata records. "
        "Resolve them before joining."
    )

# Save one metadata record per course.
(
    courses
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(
        "workspace.feedback_bronze.courses"
    )
)

display(courses.limit(10))

# COMMAND ----------

# Preview reviews with their course metadata.
display(
    spark.sql("""
    SELECT
        r.course_id,
        c.course_name,
        c.institution,
        r.rating,
        r.reviews

    FROM workspace.feedback_bronze.reviews_raw AS r

    LEFT JOIN workspace.feedback_bronze.courses AS c
        ON trim(r.course_id) = c.course_id

    LIMIT 10
    """)
)
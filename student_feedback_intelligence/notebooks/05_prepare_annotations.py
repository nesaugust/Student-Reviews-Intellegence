# Databricks notebook source
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Allow up to 200 reviews per rating: at most 1,000 in total.
REVIEWS_PER_RATING = 200

BASE_PATH = (
    "/Volumes/workspace/feedback_bronze/feedback_files/"
)

processed = spark.table(
    "workspace.feedback_gold.review_overview"
)

# Create a reproducible sample within each rating.
window = Window.partitionBy("rating").orderBy(
    F.xxhash64("review_id")
)

sample = (
    processed
    .withColumn(
        "sample_rank",
        F.row_number().over(window)
    )
    .filter(
        F.col("sample_rank") <= REVIEWS_PER_RATING
    )
    .select(
        "review_id",
        "course_id",
        "rating",
        "review_text"
    )
)

annotations = sample.toPandas()

# Keep the template available for future human annotation.
annotations["themes"] = ""

OUTPUT_PATH = BASE_PATH + "annotation_template.csv"

annotations.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("Processed reviews available:", processed.count())
print("Reviews in annotation sample:", len(annotations))

# Inspect the rating distribution.
display(
    annotations.groupby("rating")
    .size()
    .reset_index(name="sample_reviews")
)

# COMMAND ----------

# Compare saved LLM results with the current Gold table.
display(
    spark.sql("""
    SELECT
        'LLM results: themes_v1_direct' AS source,
        COUNT(DISTINCT review_id) AS reviews
    FROM workspace.feedback_gold.llm_results
    WHERE prompt_version = 'themes_v1_direct'

    UNION ALL

    SELECT
        'Gold review_overview' AS source,
        COUNT(DISTINCT review_id) AS reviews
    FROM workspace.feedback_gold.review_overview
    """)
)

# COMMAND ----------

import pandas as pd
import re
from pyspark.sql import functions as F

BASE_PATH = (
    "/Volumes/workspace/feedback_bronze/feedback_files/"
)

# Read the annotation template.
annotations = pd.read_csv(
    BASE_PATH + "annotation_template.csv",
    keep_default_na=False,
    dtype={
        "review_id": str,
        "course_id": str
    }
)

# Read existing LLM predictions for the sampled reviews only.
review_ids = annotations["review_id"].tolist()

predictions = (
    spark.table("workspace.feedback_gold.review_themes")
    .filter(F.col("review_id").isin(review_ids))
    .select(
        "review_id",
        "theme",
        "sentiment",
        "evidence"
    )
    .distinct()
    .toPandas()
)

# Combine multiple predicted themes into one annotation suggestion.
theme_lookup = (
    predictions
    .groupby("review_id")["theme"]
    .apply(lambda values: ";".join(sorted(set(values))))
    .to_dict()
)

# Keep supporting evidence for human review.
evidence_lookup = (
    predictions
    .groupby("review_id")
    .apply(
        lambda group: " | ".join(
            f"{row.theme} [{row.sentiment}]: {row.evidence}"
            for row in group.itertuples(index=False)
        )
    )
    .to_dict()
    if not predictions.empty
    else {}
)

annotations["suggested_themes"] = (
    annotations["review_id"]
    .map(theme_lookup)
    .fillna("")
)

annotations["suggested_evidence"] = (
    annotations["review_id"]
    .map(evidence_lookup)
    .fillna("")
)

# Detect Han characters, including common extension ranges.
# Han characters occur in Chinese and also in Japanese.
han_pattern = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF"
    r"\U00020000-\U0002FA1F]"
)

annotations["contains_han_characters"] = (
    annotations["review_text"]
    .apply(lambda text: bool(han_pattern.search(str(text))))
)

# Flag language cases and missing accepted predictions.
def get_review_reason(row):
    reasons = []

    if row["contains_han_characters"]:
        reasons.append("han_script_check_language")

    if not row["suggested_themes"]:
        reasons.append("no_accepted_theme_prediction")

    return ";".join(reasons)

annotations["review_reason"] = annotations.apply(
    get_review_reason,
    axis=1
)

annotations["needs_review"] = (
    annotations["review_reason"].ne("")
)

# Preserve the themes column for human labels.
# Do not copy model suggestions into the evaluation labels.
OUTPUT_PATH = BASE_PATH + "annotations_suggested.csv"

annotations.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("Saved:", OUTPUT_PATH)
print("Reviews:", len(annotations))
print(
    "Reviews containing Han characters:",
    int(annotations["contains_han_characters"].sum())
)

display(
    annotations[
        [
            "review_text",
            "suggested_themes",
            "contains_han_characters",
            "review_reason",
            "themes"
        ]
    ].head(20)
)

# COMMAND ----------

# Inspect reviews flagged for language or prediction checks.
display(
    annotations[
        annotations["needs_review"]
    ][
        [
            "review_text",
            "suggested_themes",
            "review_reason"
        ]
    ]
)
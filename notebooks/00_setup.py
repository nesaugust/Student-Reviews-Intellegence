# Databricks notebook source
# Create schemas for each processing layer.
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.feedback_bronze")
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.feedback_silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.feedback_gold")

# Create a volume for raw files and annotation files.
spark.sql("""
CREATE VOLUME IF NOT EXISTS
workspace.feedback_bronze.feedback_files
""")

print("Setup completed.")
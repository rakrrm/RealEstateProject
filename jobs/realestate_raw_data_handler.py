import sys

try:
    from awsglue.transforms import *
    from awsglue.utils import getResolvedOptions
    from pyspark.context import SparkContext
    from awsglue.context import GlueContext
    from awsglue.job import Job

    GLUE_ENV = True
except ImportError:
    GLUE_ENV = False

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower, when
from pyspark.sql.types import DoubleType

# ------------------------------------------------
# INITIALIZE SPARK / GLUE
# ------------------------------------------------
if GLUE_ENV:

    args = getResolvedOptions(sys.argv, ['JOB_NAME'])

    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session

    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)

else:

    print("Running in LOCAL MODE")

    PKGS = ",".join([
        "org.apache.hadoop:hadoop-aws:3.3.4",
        "com.amazonaws:aws-java-sdk-bundle:1.12.262"
    ])

    spark = (
        SparkSession.builder
        .appName("pyspark_test")
        .master("local[*]")
        # === S3A + credentials provider ===
        .config("spark.jars.packages", PKGS)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        # Optional: tune connections
        .config("spark.hadoop.fs.s3a.connection.maximum", "200")
        .getOrCreate()
    )

# ------------------------------------------------
# INPUT FILE
# ------------------------------------------------
if GLUE_ENV:
    s3_path = "s3://real-estate-dsr-3/landing-zone/realestate_master_2026_03_05.csv"
else:
    s3_path ="s3a://real-estate-dsr-3/landing-zone/realestate_master_2026_03_05.csv"

if GLUE_ENV:
    base_path = "s3://real-estate-dsr-3/derived-zone/"
else:
    base_path ="s3a://real-estate-dsr-3/derived-zone/"


df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(s3_path)

# ------------------------------------------------
# DATA CLEANING
# ------------------------------------------------

df = df.withColumn("org_name", trim(lower(col("org_name")))) \
    .withColumn("city", trim(lower(col("city")))) \
    .withColumn("country", trim(lower(col("country"))))

# ------------------------------------------------
# HANDLE NULL VALUES
# ------------------------------------------------

df = df.fillna({
    "org_name": "unknown",
    "city": "unknown",
    "country": "unknown",
    "rent": 0
})

# ------------------------------------------------
# VALIDATE NUMERIC VALUES
# ------------------------------------------------

df = df.withColumn("rent", col("rent").cast(DoubleType()))

df = df.withColumn(
    "rent",
    when(col("rent") < 0, 0).otherwise(col("rent"))
)

# ------------------------------------------------
# FIX INCONSISTENT COUNTRY FORMAT
# ------------------------------------------------

df = df.withColumn(
    "country",
    when(col("country") == "usa", "united states")
    .when(col("country") == "u.s.a", "united states")
    .otherwise(col("country"))
)

# ------------------------------------------------
# REMOVE DUPLICATES
# ------------------------------------------------

df = df.dropDuplicates(["property_id"])

# ------------------------------------------------
# FILTER INVALID RECORDS
# ------------------------------------------------

df = df.filter(col("property_id").isNotNull())

print("Cleaned Data")
df.show(5)

# =================================================
# CREATE DERIVED DATASETS
# =================================================

organization_df = df.select(
    "org_id",
    "org_name",
    "country"
).dropDuplicates()

portfolio_df = df.select(
    "portfolio_id",
    "portfolio_name",
    "org_id"
).dropDuplicates()

assets_df = df.select(
    "asset_id",
    "asset_name",
    "portfolio_id"
).dropDuplicates()

properties_df = df.select(
    "property_id",
    "city",
    "asset_id",
    "rent"
).dropDuplicates()

# ------------------------------------------------
# WRITE DATA AS CSV
# ------------------------------------------------

organization_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(base_path + "organization/")

portfolio_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(base_path + "portfolio/")

assets_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(base_path + "assets/")

properties_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(base_path + "properties/")

# ------------------------------------------------
# Commit Job Only in Glue
# ------------------------------------------------
if GLUE_ENV:
    job.commit()
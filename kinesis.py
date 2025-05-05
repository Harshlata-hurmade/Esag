import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

# Step 1: Init Glue Job
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Step 2: S3 input/output path (CHANGE if needed)
s3_input_path = "s3://my-kinesis-firehose-bucket-demo-123456/"  # raw Firehose data
s3_output_path = "s3://my-kinesis-firehose-bucket-demo-123456/processed/"  # output

# Step 3: Schema definition (required since auto-infer fails for JSON)
schema = StructType([
    StructField("account_hashid", StringType(), True),
    StructField("session_hashid", StringType(), True),
    StructField("consent_hashid", StringType(), True),
    StructField("page_url", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("visitor_details", StructType([
        StructField("ip_address", StringType(), True),
        StructField("ip_country", StringType(), True),
        StructField("ip_region", StringType(), True),
        StructField("city", StringType(), True),
        StructField("device_type", StringType(), True),
        StructField("timezone", StringType(), True)
    ]), True)
])

# Step 4: Read JSON from S3
df = spark.read.schema(schema).json(s3_input_path)

# Step 5: Flatten nested JSON
flat_df = df.select(
    col("account_hashid"),
    col("session_hashid"),
    col("consent_hashid"),
    col("page_url"),
    col("timestamp"),
    col("visitor_details.ip_address").alias("ip_address"),
    col("visitor_details.ip_country").alias("ip_country"),
    col("visitor_details.ip_region").alias("ip_region"),
    col("visitor_details.city").alias("city"),
    col("visitor_details.device_type").alias("device_type"),
    col("visitor_details.timezone").alias("timezone")
)

# Step 6: Convert to DynamicFrame
dyf = DynamicFrame.fromDF(flat_df, glueContext, "dyf")

# Step 7: Write to S3 in Parquet
glueContext.write_dynamic_frame.from_options(
    frame=dyf,
    connection_type="s3",
    connection_options={
        "path": s3_output_path,
        "partitionKeys": []
    },
    format="parquet"
)

print("✅ Flattened JSON data written to S3 as Parquet.")
job.commit()

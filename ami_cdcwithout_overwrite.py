from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, regexp_extract, lit, to_date,
    substring, when, current_date, upper
)
from pyspark.sql.types import *
import boto3
from datetime import datetime, timezone
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_glue_table(spark: SparkSession,
                     glue_client: boto3.client,
                     database: str,
                     table_name: str,
                     table_location: str) -> bool:
    try:
        glue_client.get_table(DatabaseName=database, Name=table_name)
        logger.info(f"Table {database}.{table_name} already exists")
        return True
    except glue_client.exceptions.EntityNotFoundException:
        try:
            schema = StructType([
                StructField("report_period_start", DateType()),
                StructField("report_period_end", DateType()),
                StructField("resource_type", StringType()),
                StructField("component_type", StringType()),
                StructField("backup_name", StringType()),
                StructField("hostname", StringType()),
                StructField("backup_type", StringType()),
                StructField("status", StringType()),
                StructField("backup_size_bytes", DoubleType()),
                StructField("load_date", DateType())
            ])

            empty_df = spark.createDataFrame([], schema)

            empty_df.write \
                .mode("append") \
                .format("parquet") \
                .option("path", table_location) \
                .saveAsTable(f"{database}.{table_name}")

            logger.info(f"Created new table {database}.{table_name} at {table_location}")
            return True
        except Exception as e:
            logger.error(f"Failed to create table: {str(e)}", exc_info=True)
            return False

def get_files(s3_client: boto3.client,
             bucket: str,
             process_date: Optional[datetime] = None) -> List[str]:
    process_date = process_date or datetime.now(timezone.utc)
    prefix = f"EC2_AMI/181565350472/ap-south-1/{process_date.strftime('%Y/%m/%d')}/"
    file_paths = []

    logger.info(f"Looking for files in: s3://{bucket}/{prefix}")

    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['Key'].lower().endswith('.csv') and obj['Size'] > 0:
                    file_paths.append(f"s3://{bucket}/{obj['Key']}")
                    logger.info(f"Found CSV file: {obj['Key']} (Size: {obj['Size']} bytes)")
        return file_paths
    except Exception as e:
        logger.error(f"Error listing files: {e}", exc_info=True)
        return []

def validate_input(df: DataFrame) -> bool:
    required_columns = {
        "Job Status", "Resource Type", "Report Time Period Start",
        "Report Time Period End", "Resource ARN", "Backup Size in Bytes",
        "Recovery Point ARN"
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    return True

def process_files(spark: SparkSession, file_paths: List[str]) -> Optional[DataFrame]:
    if not file_paths:
        logger.warning("No files provided for processing")
        return None

    try:
        df = spark.read.csv(
            file_paths,
            header=True,
            inferSchema=False,
            multiLine=True,
            escape='"'
        )

        if not validate_input(df):
            return None

        logger.info("Original DataFrame Schema:")
        df.printSchema()
        df.show(5, truncate=False)

        transformed_df = (
            df
            # Extract component_type - just "AMI" (uppercase) from "image/ami-079cdca29173a21ff"
            .withColumn("component_type",
                      when(col("Recovery Point ARN").contains("image/ami-"),
                           upper(regexp_extract(col("Recovery Point ARN"), r'image/(ami)-[^/]+', 1)))
                      .otherwise("RDS"))
            
            # Set hostname and backup_name to be the same value
            .withColumn("hostname",
                      when(col("Resource ARN").contains("instance/"),
                           regexp_extract(col("Resource ARN"), r'instance/(i-[a-f0-9]+)', 1))
                      .when(col("Resource ARN").contains("db:"),
                           regexp_extract(col("Resource ARN"), r'db:([^/]+)', 1))
                      .otherwise(col("Resource ARN")))
            .withColumn("backup_name", col("hostname"))
            
            # Status mapping
            .withColumn("status",
                      when(col("Job Status") == "COMPLETED", "SUCCESSFUL")
                      .otherwise(col("Job Status")))
            
            # Date fields
            .withColumn("report_period_start",
                      to_date(substring(col("Report Time Period Start"), 1, 10), "yyyy-MM-dd"))
            .withColumn("report_period_end",
                      to_date(substring(col("Report Time Period End"), 1, 10), "yyyy-MM-dd"))
            
            # Backup size
            .withColumn("backup_size_bytes", col("Backup Size in Bytes").cast("double"))
            
            # Load date
            .withColumn("load_date", current_date())
            
            # Backup type
            .withColumn("backup_type", lit("COMPLETE").cast(StringType()))
            
            # Resource type
            .withColumn("resource_type", col("Resource Type"))
            
            # Select final columns
            .select(
                "report_period_start",
                "report_period_end",
                "resource_type",
                "component_type",
                "backup_name",
                "hostname",
                "backup_type",
                "status",
                "backup_size_bytes",
                "load_date"
            )
        )

        logger.info("Transformed DataFrame Schema:")
        transformed_df.printSchema()
        logger.info("Sample of transformed data:")
        transformed_df.show(5, truncate=False)

        # Debug checks
        logger.info("Component type values:")
        transformed_df.select("component_type").distinct().show()
        logger.info("Hostname and backup_name values:")
        transformed_df.select("hostname", "backup_name").distinct().show(10)

        return transformed_df

    except Exception as e:
        logger.error(f"Error processing files: {str(e)}", exc_info=True)
        return None

def main():
    spark = SparkSession.builder \
        .appName("BackupReportLoader") \
        .config("spark.sql.warehouse.dir", "s3://ec2ami-backup1/glue-warehouse/") \
        .enableHiveSupport() \
        .getOrCreate()

    config = {
        "bucket": "ec2ami-backup1",
        "database": "demo_test",
        "table_name": "ec2_backup_table_test",
        "table_location": "s3://ec2ami-backup1/glue-tables/ec2_ami_backup_reports_test",
        "aws_region": "ap-south-1"
    }

    try:
        session = boto3.Session()
        s3 = session.client('s3', region_name=config["aws_region"])
        glue = session.client('glue', region_name=config["aws_region"])

        logger.info(f"Starting processing for {datetime.now(timezone.utc).date()}")

        if not create_glue_table(spark, glue, config["database"], config["table_name"], config["table_location"]):
            raise RuntimeError("Failed to create table")

        file_paths = get_files(s3, config["bucket"])
        if not file_paths:
            raise RuntimeError("No files found for processing")

        df = process_files(spark, file_paths)
        if not df:
            raise RuntimeError("Failed to process files")

        logger.info("Final data to be loaded:")
        df.show(5, truncate=False)

        logger.info(f"Writing to {config['database']}.{config['table_name']}")
        df.write \
            .mode("append") \
            .format("parquet") \
            .option("path", config["table_location"]) \
            .saveAsTable(f"{config['database']}.{config['table_name']}")

        logger.info(f"Successfully loaded {df.count()} records")

        # Verification queries
        spark.sql(f"SELECT COUNT(*) FROM {config['database']}.{config['table_name']}").show()
        spark.sql(f"""
            SELECT component_type, COUNT(*) as records
            FROM {config['database']}.{config['table_name']}
            GROUP BY component_type
        """).show()
        spark.sql(f"""
            SELECT hostname, backup_name, component_type 
            FROM {config['database']}.{config['table_name']}
            LIMIT 10
        """).show(truncate=False)

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
    finally:
        spark.stop()
        logger.info("Spark session stopped")

if __name__ == "__main__":
    main()

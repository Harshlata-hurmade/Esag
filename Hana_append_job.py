import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, lit, to_date
from pyspark.sql.types import *
import re
import boto3
from datetime import datetime
from collections import defaultdict

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Configuration
S3_BUCKET = "ec2ami-backup1"
S3_INPUT_PREFIX = "Sap_Hana/"
DATABASE_NAME = "demo_test"
TABLE_NAME = "ec2_backup_table_test"
S3_TABLE_LOCATION = f"s3://{S3_BUCKET}/data/glue_tables/{TABLE_NAME}/"

# Regular expression patterns
PATTERNS = {
    'start_date': re.compile(r"^(\d{4}-\d{2}-\d{2})"),
    'filename': re.compile(r"/usr/sap/HDB/SYS/global/hdb/backint/DB_HDB/([\w\d_-]+)"),
    'hostname': re.compile(r"state of service: .*?, (\S+):"),
    'backup_type': re.compile(r"command: BACKUP DATA FOR HDB USING"),
    'differential': re.compile(r"command: BACKUP DATA FOR HDB USING DIFFERENTIAL"),
    'success': re.compile(r"SAVE DATA finished successfully"),
    'error': re.compile(r"SAVE DATA finished with error: \[447\] backup could not be completed"),
    'size': re.compile(r"progress of service: .*?, volume: (\d+), 100% (\d+)/\d+")
}

def get_latest_backup_logs(bucket_name, prefix=""):
    """Get all .log files from S3 and return the most recent ones"""
    s3 = boto3.client('s3')
    try:
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        if 'Contents' not in response:
            raise ValueError(f"No log files found in bucket {bucket_name} with prefix {prefix}")
        
        log_files = [(obj['Key'], obj['LastModified']) 
                    for obj in response['Contents'] 
                    if obj['Key'].endswith('.log')]
        
        if not log_files:
            raise ValueError(f"No .log files found in bucket {bucket_name} with prefix {prefix}")
            
        log_files.sort(key=lambda x: x[1], reverse=True)
        return [file[0] for file in log_files]
        
    except Exception as e:
        raise Exception(f"Error accessing S3 bucket {bucket_name}: {str(e)}")

def parse_backup_content(content):
    """Parse the log file content and extract log entries"""
    backup_logs = []
    current_log = {
        "report_period_start": None,
        "report_period_end": None,
        "resource_type": "HANA",
        "component_type": "HANA",
        "backup_name": None,
        "hostname": None,
        "backup_type": "full",
        "status": "failed",
        "backup_size_bytes": 0.0,
        "load_date": datetime.now().strftime('%Y-%m-%d')
    }
    
    for line in content.splitlines():
        if "LOGBCKUP" in line:
            continue
            
        # Extract start date
        if PATTERNS['start_date'].search(line):
            current_log["report_period_start"] = PATTERNS['start_date'].search(line).group(1)
        
        # Extract filename and determine backup name
        if PATTERNS['filename'].search(line):
            filename = PATTERNS['filename'].search(line).group(1)
            if filename.endswith("_0_1"):
                continue
            
            if "databackup_differential" in filename or PATTERNS['differential'].search(line):
                current_log["backup_type"] = "differential"
                current_log["backup_name"] = current_log["report_period_start"].replace("-", "_") + "_07-30"
            else:
                current_log["backup_type"] = "full"
                current_log["backup_name"] = "COMPLETE_DATA_BACKUP"
        
        # Extract hostname
        if PATTERNS['hostname'].search(line):
            current_log["hostname"] = PATTERNS['hostname'].search(line).group(1)
        
        # Determine status and extract end_date
        if PATTERNS['success'].search(line):
            current_log["status"] = "successful"
            end_date_match = PATTERNS['start_date'].search(line)
            if end_date_match:
                current_log["report_period_end"] = end_date_match.group(1)
        elif PATTERNS['error'].search(line):
            current_log["status"] = "failed"
            end_date_match = PATTERNS['start_date'].search(line)
            if end_date_match:
                current_log["report_period_end"] = end_date_match.group(1)
        
        # Extract backup size in bytes (as double)
        if PATTERNS['size'].search(line):
            current_log["backup_size_bytes"] = float(PATTERNS['size'].search(line).group(2))
        
        # When a backup session ends, add to backup_logs
        if "SAVE DATA finished" in line and current_log["report_period_start"]:
            backup_logs.append(current_log.copy())
            
            # Reset current_log for next session
            current_log = {
                "report_period_start": None,
                "report_period_end": None,
                "resource_type": "HANA",
                "component_type": "HANA",
                "backup_name": None,
                "hostname": None,
                "backup_type": "full",
                "status": "failed",
                "backup_size_bytes": 0.0,
                "load_date": datetime.now().strftime('%Y-%m-%d')
            }
    
    return backup_logs

def process_backup_logs(bucket_name, file_keys):
    """Download and process multiple log files from S3"""
    s3 = boto3.client('s3')
    all_backup_data = []
    
    for file_key in file_keys:
        try:
            print(f"Processing log file: {file_key}")
            obj = s3.get_object(Bucket=bucket_name, Key=file_key)
            content = obj['Body'].read().decode('utf-8')
            backup_data = parse_backup_content(content)
            all_backup_data.extend(backup_data)
        except Exception as e:
            print(f"Warning: Error processing file {file_key}: {str(e)}")
            continue
    
    return all_backup_data

def save_to_glue_catalog(data, database_name, table_name, table_location):
    """Save processed results to Glue Data Catalog with proper schema handling"""
    # Create initial DataFrame from raw data
    initial_df = spark.createDataFrame(data)
    
    # Define the target schema
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
    
    # Convert date fields and ensure proper types
    final_df = initial_df.select(
        to_date(col("report_period_start"), "yyyy-MM-dd").alias("report_period_start"),
        to_date(col("report_period_end"), "yyyy-MM-dd").alias("report_period_end"),
        col("resource_type"),
        col("component_type"),
        col("backup_name"),
        col("hostname"),
        col("backup_type"),
        col("status"),
        col("backup_size_bytes").cast("double"),
        to_date(col("load_date"), "yyyy-MM-dd").alias("load_date")
    )
    
    # Check if table exists
    table_exists = spark.catalog.tableExists(f"{database_name}.{table_name}")
    
    if table_exists:
        print(f"Table exists, appending new data to {database_name}.{table_name}")
        
        # Create temporary view of new data
        temp_view_name = f"temp_view_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        final_df.createOrReplaceTempView(temp_view_name)
        
        # Append data to the existing table
        spark.sql(f"""
            INSERT INTO TABLE {database_name}.{table_name}
            SELECT * FROM {temp_view_name}
        """)
        
        # Clean up
        spark.catalog.dropTempView(temp_view_name)
        
        print("New data successfully appended to table")
    else:
        print(f"Creating new table {database_name}.{table_name}")
        final_df.write.format("parquet") \
            .mode("append") \
            .option("path", table_location) \
            .saveAsTable(f"{database_name}.{table_name}")
    
    print(f"Successfully saved {final_df.count()} records to {database_name}.{table_name}")

def main():
    try:
        # Get and process all backup log files
        log_files = get_latest_backup_logs(S3_BUCKET, S3_INPUT_PREFIX)
        print(f"Found {len(log_files)} log files to process")
        
        backup_data = process_backup_logs(S3_BUCKET, log_files)
        if not backup_data:
            raise ValueError("No valid backup entries found in log files")
        
        # Save results to Glue Catalog
        save_to_glue_catalog(backup_data, DATABASE_NAME, TABLE_NAME, S3_TABLE_LOCATION)
        
        job.commit()
        print("Job completed successfully")
    except Exception as e:
        print(f"Job failed: {str(e)}")
        job.commit()
        raise e

if __name__ == "__main__":
    main()

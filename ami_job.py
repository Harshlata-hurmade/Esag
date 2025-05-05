from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, round, regexp_extract, lit, to_date, 
    substring, when, max as sql_max, current_date
)
from pyspark.sql.types import *
import boto3
from botocore.exceptions import ClientError
import time
from datetime import datetime, timedelta, timezone

def get_todays_folder(s3_client, bucket, base_prefix):
    """Get today's date folder path in YYYY/MM/DD format"""
    today = datetime.now(timezone.utc)
    todays_prefix = f"{base_prefix}{today.strftime('%Y/%m/%d/')}"
    
    print(f"Looking for today's folder: {todays_prefix}")
    
    try:
        # Check if folder exists by listing at least one object
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=todays_prefix,
            MaxKeys=1
        )
        if response.get('KeyCount', 0) > 0:
            print(f"Today's folder exists: {todays_prefix}")
            return todays_prefix
        else:
            print(f"Today's folder is empty or doesn't exist: {todays_prefix}")
            return None
    except ClientError as e:
        print(f"Error checking today's folder: {e}")
        return None

def find_todays_report_files(s3_client, bucket, todays_prefix):
    """Find all report files in today's folder"""
    report_files = []
    
    print(f"\nScanning for report files in: {todays_prefix}")
    
    try:
        # Find all backup_jobs_report_* folders in today's folder
        report_paginator = s3_client.get_paginator('list_objects_v2')
        for report_page in report_paginator.paginate(
            Bucket=bucket,
            Prefix=todays_prefix,
            Delimiter='/'
        ):
            for report_folder in report_page.get('CommonPrefixes', []):
                if 'backup_jobs_report_' in report_folder['Prefix'].lower():
                    print(f"Found report folder: {report_folder['Prefix']}")
                    
                    # Find all CSV files in this report folder
                    file_paginator = s3_client.get_paginator('list_objects_v2')
                    for file_page in file_paginator.paginate(
                        Bucket=bucket,
                        Prefix=report_folder['Prefix']
                    ):
                        for file in file_page.get('Contents', []):
                            file_key = file['Key']
                            if (file_key.lower().endswith('.csv') and 
                               'backup_job_report_' in file_key.lower()):
                                
                                s3_path = f"s3://{bucket}/{file_key}"
                                report_files.append({
                                    'path': s3_path,
                                    'last_modified': file.get('LastModified'),
                                    'size': file.get('Size', 0)
                                })
                                print(f"Found file: {file_key}")
    
    except ClientError as e:
        print(f"Error processing folder {todays_prefix}: {e}")
    
    print(f"\nTotal report files found today: {len(report_files)}")
    return report_files

def process_files_to_glue(spark, file_infos, glue_database, glue_table, table_location, overwrite=False):
    """Process files and load data to Glue table with controlled append/overwrite"""
    print("\nStarting processing to Glue table...")
    
    try:
        # Get just the paths for reading
        file_paths = [f['path'] for f in file_infos]
        
        # Read files with header
        df = spark.read.csv(
            file_paths,
            header=True,
            inferSchema=False,
            multiLine=True,
            escape='"'
        )
        
        # Select and transform required columns
        required_columns = [
            "Report Time Period Start",
            "Report Time Period End",
            "Job Status",
            "Resource Type", 
            "Resource ARN",
            "Backup Size in Bytes"
        ]
        
        # Handle missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns in source data: {missing_columns}")
            existing_columns = [col for col in required_columns if col in df.columns]
            result_df = df.select(*existing_columns)
        else:
            result_df = df.select(*required_columns)
        
        # Transformations
        result_df = (result_df
            .withColumnRenamed("Job Status", "job_status")
            .withColumnRenamed("Resource Type", "resource_type")
            .withColumn("job_status", 
                when(col("job_status") == "COMPLETED", "SUCCESSFUL").otherwise(col("job_status")))
            .withColumn("report_period_start", 
                to_date(substring(col("Report Time Period Start"), 1, 10), "yyyy-MM-dd"))
            .withColumn("report_period_end", 
                to_date(substring(col("Report Time Period End"), 1, 10), "yyyy-MM-dd"))
            .withColumn("resource_id",
                when(col("Resource ARN").contains("instance/"),
                    regexp_extract(col("Resource ARN"), r'instance/(i-[a-f0-9]+)', 1))
                .when(col("Resource ARN").contains("db:"),
                    regexp_extract(col("Resource ARN"), r'db:([^/]+)', 1))
                .otherwise(col("Resource ARN")))
            .withColumn("size_gb",
                round(col("Backup Size in Bytes").cast("double") / (1024 * 1024 * 1024), 2))
            .withColumn("backup_type", lit("full"))
            .withColumn("file_last_modified", lit(datetime.now(timezone.utc)).cast("timestamp"))
            .withColumn("load_date", current_date())
            .drop("Report Time Period Start", "Report Time Period End", "Backup Size in Bytes", "Resource ARN")
        )
        
        # Show sample data
        print("Transformed data sample:")
        result_df.show(5, truncate=False)
        
        # Initialize Glue client and update table
        glue_client = boto3.client('glue')
        if not create_or_update_glue_table(glue_client, glue_database, glue_table, table_location):
            print("Failed to create/update Glue table")
            return False
        
        # Determine write mode
        write_mode = "overwrite" if overwrite else "append"
        print(f"\nWriting data with mode: {write_mode}")
        
        # Write data
        (result_df.write
            .mode(write_mode)
            .format("parquet")
            .option("path", table_location)
            .saveAsTable(f"{glue_database}.{glue_table}"))
        
        print(f"Successfully processed {result_df.count()} records")
        
        # Refresh and verify
        spark.sql(f"REFRESH TABLE {glue_database}.{glue_table}")
        print("Verification queries:")
        spark.sql(f"SELECT COUNT(*) FROM {glue_database}.{glue_table}").show()
        spark.sql(f"""
            SELECT date(file_last_modified) as load_date, COUNT(*) as records 
            FROM {glue_database}.{glue_table}
            GROUP BY 1 ORDER BY 1 DESC
        """).show()
        
        return True
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        return False

def main():
    spark = SparkSession.builder \
        .appName("TodayBackupReportProcessor") \
        .config("spark.hadoop.fs.s3.maxRetries", "20") \
        .config("spark.hadoop.fs.s3.retryIntervalSeconds", "10") \
        .config("spark.sql.warehouse.dir", "s3://ec2ami-backup1/glue-warehouse/") \
        .enableHiveSupport() \
        .getOrCreate()
    
    s3 = boto3.client('s3', config=boto3.session.Config(
        retries={'max_attempts': 10, 'mode': 'standard'}
    ))
    
    # Configuration
    glue_database = "demo_test"
    glue_table = "ec2_ami_backup_test2"
    table_location = "s3://ec2ami-backup1/glue-tables/ec2_ami_backup_reports_test"
    base_path = "s3://ec2ami-backup1/EC2_AMI/181565350472/ap-south-1"
    bucket = base_path.split('/')[2]
    prefix = '/'.join(base_path.split('/')[3:]) + '/'
    
    print(f"Starting daily processing for {datetime.now(timezone.utc).date()}")
    
    # Step 1: Get today's folder
    todays_prefix = get_todays_folder(s3, bucket, prefix)
    if not todays_prefix:
        print("No today's folder found. Exiting.")
        spark.stop()
        return
    
    # Step 2: Find today's report files
    file_infos = find_todays_report_files(s3, bucket, todays_prefix)
    if not file_infos:
        print("No report files found for today. Exiting.")
        spark.stop()
        return
    
    # Step 3: Process files (set overwrite=False for append mode)
    start_time = time.time()
    success = process_files_to_glue(
        spark, 
        file_infos, 
        glue_database, 
        glue_table, 
        table_location,
        overwrite=False  # Set to True if you want to replace existing data
    )
    
    if success:
        print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")
    else:
        print("\nProcessing failed")
    
    spark.stop()

if __name__ == "__main__":
    main()

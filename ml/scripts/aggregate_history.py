
import os
import pandas as pd
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from ml.utils.logger import logger
from ml.utils.s3_utils import download_s3_folder, upload_file_to_s3

def aggregate_prediction_history(
    bucket_name: str,
    s3_prefix: str = "predictions/",
    output_file: str = "prediction_history_master.csv",
    temp_dir: str = "temp_history",
    aws_region: Optional[str] = "us-east-1"
) -> Optional[str]:
    """
    Downloads all prediction files from S3, aggregates them, and saves a master CSV.
    
    Args:
        bucket_name: S3 Bucket name
        s3_prefix: Prefix where individual prediction files are stored
        output_file: Name of the aggregated output file
        temp_dir: Local temporary directory for downloads
        aws_region: AWS Region
        
    Returns:
        Path to the local aggregated file, or None if failed.
    """
    logger.info("=" * 60)
    logger.info("⏳ STARTING PREDICTION HISTORY AGGREGATION")
    logger.info("=" * 60)
    
    # 1. Download Files
    logger.info(f"Downloading from s3://{bucket_name}/{s3_prefix} to {temp_dir}...")
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
        
    success = download_s3_folder(bucket_name, s3_prefix, temp_dir, aws_region)
    if not success:
        logger.error("Failed to download files from S3.")
        return None
        
    # 2. Iterate and Aggregate
    all_dfs = []
    
    # Regex to extract metadata from filename: predictions_<HASH>_<TIMESTAMP>.csv
    # Example: predictions_abc1234_20231027123045.csv
    filename_pattern = re.compile(r"predictions_([a-f0-9]+)_(\d{14})\.csv")
    
    csv_files = list(Path(temp_dir).glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files to aggregate.")
    
    if not csv_files:
        logger.warning("No CSV files found.")
        return None
        
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # Extract Metadata
            match = filename_pattern.search(file_path.name)
            if match:
                commit_hash = match.group(1)
                timestamp_str = match.group(2)
                
                # Convert timestamp string to ISO format for readability
                try:
                    dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                    iso_timestamp = dt.isoformat()
                except ValueError:
                    iso_timestamp = timestamp_str
                
                # Add columns
                df["commit_hash"] = commit_hash
                df["timestamp"] = iso_timestamp
                df["source_file"] = file_path.name
                
            else:
                # Fallback for files matching generic 'predictions.csv' or other formats
                df["commit_hash"] = "unknown"
                df["timestamp"] = datetime.now().isoformat()
                df["source_file"] = file_path.name
                logger.warning(f"Could not parse metadata from filename: {file_path.name}")
                
            all_dfs.append(df)
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            
    # 3. Concatenate
    if not all_dfs:
        logger.warning("No valid data found to aggregate.")
        return None
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"✓ Aggregated {len(master_df)} total rows from {len(all_dfs)} files.")
    
    # 4. Save Locally
    master_df.to_csv(output_file, index=False)
    logger.info(f"💾 Saved master file to: {output_file}")
    
    # 5. Upload Master to S3
    history_prefix = "history"
    logger.info(f"🚀 Uploading to s3://{bucket_name}/{history_prefix}/{output_file}...")
    
    upload_success = upload_file_to_s3(
        local_path=output_file,
        bucket_name=bucket_name,
        s3_prefix=history_prefix,
        object_name=output_file,
        aws_region=aws_region
    )
    
    # Cleanup
    # import shutil
    # shutil.rmtree(temp_dir)
    # if os.path.exists(output_file):
    #    os.remove(output_file)
        
    if upload_success:
        logger.info("✅ Aggregation and Upload Complete!")
        return output_file
    else:
        logger.error("❌ Upload failed.")
        return None

if __name__ == "__main__":
    # Test run (requires env vars)
    bucket = os.getenv("S3_BUCKET")
    if bucket:
        aggregate_prediction_history(bucket)
    else:
        print("S3_BUCKET not set. Skipping run.")

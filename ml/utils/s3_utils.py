
import os
import logging
from pathlib import Path
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None

logger = logging.getLogger(__name__)

def get_s3_client(aws_region: Optional[str] = None):
    """Create and return an S3 client."""
    if not boto3:
        logger.error("boto3 is not installed. Please install it: pip install boto3")
        return None
        
    try:
        if aws_region:
            return boto3.client('s3', region_name=aws_region)
        return boto3.client('s3')
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        return None

def download_s3_folder(bucket_name: str, s3_prefix: str, local_dir: str, aws_region: Optional[str] = None) -> bool:
    """
    Download all files from an S3 bucket prefix to a local directory.
    """
    s3_client = get_s3_client(aws_region)
    if not s3_client:
        return False
    
    # Ensure local directory exists
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Remove trailing slash from prefix if present
    s3_prefix = s3_prefix.rstrip('/')
    
    try:
        logger.info(f"Listing objects in s3://{bucket_name}/{s3_prefix}")
        
        # List all objects with the given prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
        
        file_count = 0
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                s3_key = obj['Key']
                
                # Skip if it's just a folder marker
                if s3_key.endswith('/'):
                    continue
                
                # Calculate relative path and local file path
                if s3_prefix:
                    # If prefix is "models", we want "models/file.txt" -> "file.txt"
                    # But if prefix is "models/", s3_key is "models/file.txt" logic needs to be robust
                    # Using string replacement/substring might be safer specifically for prefix
                    if s3_key.startswith(s3_prefix):
                        relative_path = s3_key[len(s3_prefix):].lstrip('/')
                    else:
                        relative_path = s3_key
                else:
                    relative_path = s3_key
                
                local_file = local_path / relative_path
                
                # Create parent directories if needed
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Download the file
                logger.debug(f"Downloading: {s3_key} -> {local_file}")
                s3_client.download_file(bucket_name, s3_key, str(local_file))
                file_count += 1
        
        logger.info(f"Successfully downloaded {file_count} files from S3")
        return True
        
    except ClientError as e:
        logger.error(f"AWS S3 error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def upload_file_to_s3(local_path: str, bucket_name: str, s3_prefix: str, object_name: Optional[str] = None, aws_region: Optional[str] = None) -> bool:
    """
    Upload a single file to S3.
    
    Args:
        local_path: Path to the local file
        bucket_name: Name of the S3 bucket
        s3_prefix: Destination folder/prefix in S3 (e.g., 'models/trained_models/')
        object_name: Optional custom name for the file in S3. If None, uses local filename.
        aws_region: AWS region
        
    Returns:
        bool: True if successful
    """
    s3_client = get_s3_client(aws_region)
    if not s3_client:
        return False
        
    path = Path(local_path)
    if not path.exists():
        logger.error(f"Local file not found: {local_path}")
        return False
        
    # Construct S3 key
    s3_prefix = s3_prefix.rstrip('/')
    
    # Use provided object name or valid local filename
    target_name = object_name if object_name else path.name
    s3_key = f"{s3_prefix}/{target_name}"
    
    try:
        logger.info(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
        s3_client.upload_file(str(local_path), bucket_name, s3_key)
        logger.info("Upload successful")
        return True
    except ClientError as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        return False

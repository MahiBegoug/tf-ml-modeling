#!/usr/bin/env python3
"""
Download pre-trained models from AWS S3 bucket.

This script downloads model files from S3 to the local filesystem.
It can be configured via environment variables or command-line arguments.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("ERROR: boto3 is not installed. Please install it: pip install boto3")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_s3_folder(bucket_name, s3_prefix, local_dir, aws_region=None):
    """
    Download all files from an S3 bucket prefix to a local directory.
    
    Args:
        bucket_name: Name of the S3 bucket
        s3_prefix: Prefix path in S3 (e.g., 'models/' or 'models/trained_models/')
        local_dir: Local directory to download files to
        aws_region: AWS region (optional, uses default if not specified)
    """
    # Create S3 client
    if aws_region:
        s3_client = boto3.client('s3', region_name=aws_region)
    else:
        s3_client = boto3.client('s3')
    
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
                logger.warning(f"No objects found with prefix: {s3_prefix}")
                continue
                
            for obj in page['Contents']:
                s3_key = obj['Key']
                
                # Skip if it's just a folder marker
                if s3_key.endswith('/'):
                    continue
                
                # Calculate relative path and local file path
                if s3_prefix:
                    relative_path = s3_key[len(s3_prefix):].lstrip('/')
                else:
                    relative_path = s3_key
                
                local_file = local_path / relative_path
                
                # Create parent directories if needed
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Download the file
                logger.info(f"Downloading: {s3_key} -> {local_file}")
                s3_client.download_file(bucket_name, s3_key, str(local_file))
                file_count += 1
        
        logger.info(f"Successfully downloaded {file_count} files from S3")
        return True
        
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please configure AWS credentials.")
        logger.error("You can set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        return False
    except ClientError as e:
        logger.error(f"AWS S3 error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download pre-trained models from AWS S3'
    )
    parser.add_argument(
        '--bucket',
        default=os.getenv('S3_BUCKET', ''),
        help='S3 bucket name (or set S3_BUCKET env var)'
    )
    parser.add_argument(
        '--prefix',
        default=os.getenv('S3_MODEL_PREFIX', 'models/'),
        help='S3 prefix/folder path (or set S3_MODEL_PREFIX env var)'
    )
    parser.add_argument(
        '--output',
        default=os.getenv('LOCAL_MODEL_DIR', '/app/pre_trained_defect_models'),
        help='Local output directory (or set LOCAL_MODEL_DIR env var)'
    )
    parser.add_argument(
        '--region',
        default=os.getenv('AWS_REGION', 'us-east-1'),
        help='AWS region (or set AWS_REGION env var)'
    )
    parser.add_argument(
        '--skip-if-exists',
        action='store_true',
        help='Skip download if local directory already has files'
    )
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.bucket:
        logger.error("S3 bucket name is required. Use --bucket or set S3_BUCKET environment variable.")
        sys.exit(1)
    
    # Check if we should skip download
    if args.skip_if_exists:
        local_path = Path(args.output)
        if local_path.exists() and any(local_path.iterdir()):
            logger.info(f"Local directory {args.output} already contains files. Skipping download.")
            sys.exit(0)
    
    logger.info("=" * 60)
    logger.info("S3 Model Download Configuration:")
    logger.info(f"  Bucket: {args.bucket}")
    logger.info(f"  Prefix: {args.prefix}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Region: {args.region}")
    logger.info("=" * 60)
    
    # Download models
    success = download_s3_folder(
        bucket_name=args.bucket,
        s3_prefix=args.prefix,
        local_dir=args.output,
        aws_region=args.region
    )
    
    if success:
        logger.info("Model download completed successfully!")
        sys.exit(0)
    else:
        logger.error("Model download failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()

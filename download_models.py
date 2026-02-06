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
from ml.utils.s3_utils import download_s3_folder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

"""
Simple ML Runner
===============

Easy entry point for Training and Prediction.
Configuration is loaded from 'ml_config.json'.

Usage:
    python run_ml.py train --csv data.csv --target defect --model randomforest
    python run_ml.py predict --csv new_data.csv --model my_model
"""

import argparse
import json
import os
import sys

# Ensure we can import infrastructure
sys.path.append(os.getcwd())

from ml.services.incremental_learning_service import IncrementalLearningService
from ml.services.user_training_service import UserTrainingService
from ml.scripts.aggregate_history import aggregate_prediction_history
from ml.utils.logger import logger
import pandas as pd
import numpy as np

def load_config(path="ml_config.json"):
    if not os.path.exists(path):
        print(f"Config file not found: {path}. Using defaults.")
        return {
            "directories": {"models": "models", "features": "features"},
            "defaults": {"model_type": "randomforest", "n_jobs": 1}
        }
    with open(path, "r") as f:
        return json.load(f)

def cmd_train(args, config):
    print("="*50)
    print("MODE: RETRAINING ON FULL HISTORY")
    print("="*50)
    
    classes_weights = config.get("defaults", {}).get("class_weight", None) # Optional global default
    
    # Check for custom model params
    custom_params = config.get("model_params", {})
    if not custom_params:
         # Check inside defaults if not at root
         custom_params = config.get("defaults", {}).get("model_params", {})
         
    metrics = service.train_from_csv(
        csv_path=args.csv,
        target_col=args.target,
        model_type=args.model,
        top_k_features=defaults.get("top_k_features", 20),
        tune_hyperparameters=tune,
        tune_n_jobs=defaults.get("n_jobs", 1),
        output_name=args.name,
        custom_params=custom_params
    )
    
    print("\n✅ Full Retraining Complete!")
    print(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
    print(f"Model saved to: {metrics.get('model_path')}")

def cmd_predict(args, config):
    print("="*50)
    print("MODE: TEST DIRECTLY")
    print("="*50)

    dirs = config.get("directories", {})
    defaults = config.get("defaults", {})
    
    service = UserTrainingService(
        models_dir=dirs.get("models", "models"),
        features_dir=dirs.get("features", "features")
    )
    
    # --- Auto-Download Models if Missing ---
    model_name = args.model
    # Check if model exists locally
    # Service expects: {models_dir}/{model_name}.joblib
    model_path = os.path.join(service.models_dir, f"{model_name}.joblib")
    
    if not os.path.exists(model_path):
        s3_bucket = os.getenv('S3_BUCKET')
        if s3_bucket:
            print(f"⚠️ Model '{model_name}' not found locally at {model_path}")
            print("🚀 S3_BUCKET detected. Attempting to download models from S3...")
            from ml.utils.s3_utils import download_s3_folder
            
            s3_prefix = os.getenv('S3_MODEL_PREFIX', 'models/')
            
            # CRITICAL FIX: 
            # 1. S3 structure includes 'trained_models/' (e.g., prefix/trained_models/model.joblib)
            # 2. service.models_dir likely ends in 'trained_models' (e.g., /app/models/trained_models)
            # 3. If we download TO service.models_dir, we get /app/models/trained_models/trained_models/... (Double Nesting)
            # 4. So we must download to the PARENT of service.models_dir
            
            # We want to flatten the structure if possible, or at least control it.
            # If we set s3_prefix to ".../trained_models/", we get the files directly.
            # So we should download into service.models_dir.
            download_target = service.models_dir
            
            download_s3_folder(
                bucket_name=s3_bucket,
                s3_prefix=s3_prefix,
                local_dir=download_target,
                aws_region=os.getenv('AWS_REGION', 'us-east-1')
            )
        else:
            print(f"⚠️ Model '{model_name}' not found and S3_BUCKET not set. Prediction may fail.")

    output_csv = args.output
    if not output_csv:
        output_csv = f"predictions_{os.path.basename(args.csv)}"
        
    # Get Threshold
    threshold = config.get("execution", {}).get("threshold")
    if hasattr(args, 'threshold') and args.threshold is not None:
        threshold = args.threshold
        
    if threshold is None:
        threshold = defaults.get("threshold", 0.5)
        
    results = service.predict_from_csv(
        input_csv=args.csv,
        model_name=args.model,
        output_csv=output_csv,
        threshold=threshold
    )
    
    print("\n✅ Direct Testing Complete!")
    print(f"Saved to: {output_csv}")
    print(results.head())
    
    # Auto-upload to S3 if configured
    s3_bucket = os.getenv('S3_BUCKET')
    if s3_bucket:
        from ml.utils.s3_utils import upload_file_to_s3
        s3_prefix = os.getenv('S3_MODEL_PREFIX', 'models/').rstrip('/')
        
        # Generate Versioned Filename: predictions_<COMMIT>_<TIMESTAMP>.csv
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        commit_hash = os.getenv('GITHUB_SHA', 'manual')[:7]  # Short hash
        
        base_name = os.path.basename(output_csv)
        name_stem = os.path.splitext(base_name)[0]
        ext = os.path.splitext(base_name)[1]
        
        versioned_name = f"{name_stem}_{commit_hash}_{timestamp}{ext}"
        target_key = f"predictions/{versioned_name}"
        
        print(f"🚀 Uploading predictions to s3://{s3_bucket}/{target_key}...")
        upload_file_to_s3(
            local_path=output_csv,
            bucket_name=s3_bucket,
            s3_prefix="predictions", 
            object_name=versioned_name,
            aws_region=os.getenv('AWS_REGION', 'us-east-1')
        )
        print("✓ Upload Complete")

def cmd_incremental(args, config):
    print("="*50)
    print("MODE: INCREMENTAL LEARNING")
    print("="*50)
    
    dirs = config.get("directories", {})
    
    service = IncrementalLearningService(
        models_dir=dirs.get("models", "models")
    )
    
    # Load Data
    if hasattr(args, 'raw_data') and args.raw_data:
        print(f"   Loading {len(args.raw_data)} samples from Config...")
        df = pd.DataFrame(args.raw_data)
    else:
        print(f"   Loading data from {args.csv}...")
        df = pd.read_csv(args.csv)
        
    target = args.target
    
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found")
        
    # Prepare Dicts
    # Assumption: Feature columns are everything except target
    # Incremental Service expects aligned feature vectors (matching existing model order)
    # We'll use row indices as fake block_ids
    
    feature_vectors = {}
    labels = {}
    
    feature_cols = [c for c in df.columns if c != target]
    
    for i, row in df.iterrows():
        feature_vectors[f"inc_row_{i}"] = row[feature_cols].values.tolist()
        labels[f"inc_row_{i}"] = int(row[target])
        
    metrics = service.incremental_update(
        model_name=args.model,
        new_features=feature_vectors,
        new_labels=labels
    )
    
    print("\n✅ Incremental Update Complete!")
    print(f"Method Used: {metrics.get('update_method')}")
    print(f"MCC on New Data: {metrics.get('mcc', 'N/A')}")

def cmd_aggregate(args, config):
    print("="*50)
    print("MODE: HISTORY AGGREGATION")
    print("="*50)
    
    s3_bucket = os.getenv("S3_BUCKET")
    if not s3_bucket:
        print("❌ S3_BUCKET environment variable not set.")
        return
        
    output_file = args.output
    if not output_file:
        output_file = "prediction_history_master.csv"
        
    print(f"Aggregating history from s3://{s3_bucket}/predictions/...")
    
    result = aggregate_prediction_history(
        bucket_name=s3_bucket,
        s3_prefix=args.prefix,
        output_file=output_file,
        aws_region=os.getenv("AWS_REGION", "us-east-1")
    )
    
    if result:
        print(f"✅ Aggregation Complete! Master file saved to: {result}")
        print(f"   Also uploaded to s3://{s3_bucket}/history/{output_file}")
    else:
        print("❌ Aggregation Failed. Check logs.")

def run_from_config(config):
    """Execute based purely on config file."""
    exec_config = config.get("execution", {})
    mode = exec_config.get("mode", "train").lower()
    
    print(f"[Run] Running from JSON Config | Mode: {mode.upper()}")
    
    # Map config generic keys to specific args needed by functions
    class Args: pass
    args = Args()
    args.csv = exec_config.get("csv_path")
    args.raw_data = exec_config.get("data") # Support direct data entry
    args.target = exec_config.get("target_col", "defect")
    args.output = exec_config.get("output_path")
    args.tune = config.get("defaults", {}).get("tune_hyperparameters", False)
    
    # Handle Model Name vs Type
    if mode in ["train", "retrain"]:
        # For training, we need TYPE (algorithm) and NAME (output file)
        # Check if user specified a specific type in execution, else use default
        args.model = exec_config.get("model_type", config.get("defaults", {}).get("model_type", "randomforest"))
        args.name = exec_config.get("model_name")
    else:
        # For predict/incremental, 'model' arg is the NAME of the saved model
        args.model = exec_config.get("model_name")

    # Validate
    if not args.csv and not args.raw_data:
        logger.error("❌ Both 'csv_path' and 'data' are missing in ml_config.json execution section.")
        return

    try:
        if mode in ["train", "retrain"]:
            cmd_train(args, config)
        elif mode in ["predict", "test"]:
            cmd_predict(args, config)
        elif mode in ["incremental", "update"]:
            cmd_incremental(args, config)
        else:
            logger.error(f"❌ Unknown mode '{mode}' in ml_config.json. Use: train, predict, incremental")
    except Exception as e:
        logger.error(f"Execution failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="ML Runner")
    config = load_config()
    defaults = config.get("defaults", {})
    
    # Make subcommands optional
    subparsers = parser.add_subparsers(dest="command", required=False)
    
    # Train Command
    train_parser = subparsers.add_parser("train", help="Train from scratch (Full History)")
    train_parser.add_argument("--csv", required=True, help="Input CSV file")
    train_parser.add_argument("--target", default="defect", help="Target column name")
    train_parser.add_argument("--model", default=defaults.get("model_type", "randomforest"), 
                              help="Model type")
    train_parser.add_argument("--name", help="Output model name")
    train_parser.add_argument("--tune", action="store_true", help="Enable tuning")
    
    # Prediction Command
    pred_parser = subparsers.add_parser("predict", help="Test directly using a model")
    pred_parser.add_argument("--csv", required=True, help="Input CSV file")
    pred_parser.add_argument("--model", required=True, help="Model name")
    pred_parser.add_argument("--output", help="Output CSV path")
    pred_parser.add_argument("--threshold", type=float, help="Classification threshold (default: config or 0.5)")
    
    # Incremental Command
    inc_parser = subparsers.add_parser("incremental", help="Update existing model (Incremental Learning)")
    inc_parser.add_argument("--csv", required=True, help="New data CSV")
    inc_parser.add_argument("--target", default="defect", help="Target column name")
    inc_parser.add_argument("--model", required=True, help="Model name to update")
    
    # Aggregation Command
    agg_parser = subparsers.add_parser("aggregate", help="Aggregate prediction history from S3")
    agg_parser.add_argument("--prefix", default="predictions/", help="S3 prefix to search (default: predictions/)")
    agg_parser.add_argument("--output", help="Output filename (default: prediction_history_master.csv)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        cmd_train(args, config)
    elif args.command == "predict":
        cmd_predict(args, config)
    elif args.command == "incremental":
        cmd_incremental(args, config)
    elif args.command == "aggregate":
        cmd_aggregate(args, config)
    else:
        # No command argument -> Use Config
        run_from_config(config)

if __name__ == "__main__":
    main()

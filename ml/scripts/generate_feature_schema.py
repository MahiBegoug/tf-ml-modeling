import os
import joblib
import pandas as pd
from ml.utils.logger import logger
from ml.utils.s3_utils import download_s3_folder

def generate_schema(model_name="randomforest_model"):
    # 1. Ensure models are present (download if needed)
    models_dir = "models"
    features_dir = "features"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    
    if not os.path.exists(model_path):
        logger.info(f"Downloading model {model_name} from S3...")
        s3_bucket = os.getenv("S3_BUCKET")
        if not s3_bucket:
            logger.error("S3_BUCKET not set.")
            return

        download_s3_folder(
            bucket_name=s3_bucket,
            s3_prefix="pre_trained_defect_models/trained_models/",
            local_dir=models_dir,
            aws_region="us-east-1"
        )
    
    if not os.path.exists(model_path):
        logger.error(f"Failed to find model at {model_path}")
        return

    # 2. Load Model
    try:
        logger.info(f"Loading model from {model_path}...")
        bundle = joblib.load(model_path)
        model = bundle['model'] if isinstance(bundle, dict) else bundle
        
        # 3. Extract Features
        if hasattr(model, "feature_names_in_"):
            features = model.feature_names_in_.tolist()
            logger.info(f"Found {len(features)} features in model.")
            
            # 4. Save Schema
            schema_path = os.path.join(features_dir, f"{model_name}_features.csv")
            with open(schema_path, "w") as f:
                f.write("\n".join(features))
            logger.info(f"✅ Schema saved to {schema_path}")
            
        else:
            logger.error("Model does not have 'feature_names_in_' attribute.")
            
    except Exception as e:
        logger.error(f"Error processing model: {e}")

if __name__ == "__main__":
    generate_schema()

"""
Model Saver Utility
==================

Standardizes model saving to ensure compatibility with PredictionService.
Saves models as bundles containing:
- The trained model
- The scaler (optional but recommended)
- Metadata (model type, metrics, timestamp)
- Feature information (names, valid counts)
"""

import os
import joblib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from ml.utils.logger import logger

class ModelSaver:
    """
    Handles saving models in a standardized 'bundle' format.
    """
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

    def save_model(self,
                   model: Any,
                   model_name: str,
                   scaler: Optional[Any] = None,
                   feature_names: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a model bundle to disk.
        
        Args:
            model: The trained model object
            model_name: Name of the model (without .joblib extension)
            scaler: Fitted scaler (optional)
            feature_names: List of feature names in order (optional)
            metadata: Any additional metadata (metrics, params, etc.)
            
        Returns:
            Absolute path to the saved model file
        """
        # Ensure .joblib extension
        if model_name.endswith('.joblib'):
            filename = model_name
        else:
            filename = f"{model_name}.joblib"
            
        save_path = os.path.join(self.models_dir, filename)
        
        # Prepare bundle
        bundle = {
            'model': model,
            'scaler': scaler,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Add feature info if available
        if feature_names:
            bundle['feature_order'] = feature_names
            bundle['n_features'] = len(feature_names)
            
        try:
            joblib.dump(bundle, save_path)
            logger.info(f"✓ Saved model bundle to: {save_path}")
            if feature_names:
                logger.info(f"  - Features: {len(feature_names)}")
            if scaler:
                logger.info(f"  - Scaler: {type(scaler).__name__}")
                
            # --- Check for S3 Auto-Upload ---
            s3_bucket = os.getenv('S3_BUCKET')
            if s3_bucket:
                from ml.utils.s3_utils import upload_file_to_s3
                
                # Determine prefix (folder)
                # By default, we might upload to 'pre_trained_defect_models/trained_models/'
                # But here we just want to mimic the structure.
                # Let's assume S3_MODEL_PREFIX points to root (e.g. pre_trained_defect_models/)
                # And we want to put models in 'trained_models' subfolder if that matches local structure?
                # Or just use the S3_MODEL_PREFIX as the folder.
                
                # Simple logic: Upload to S3_MODEL_PREFIX/filename
                s3_prefix = os.getenv('S3_MODEL_PREFIX', 'models/')
                # Ensure it points to 'trained_models' if that's the convention?
                # The user's S3 has 'pre_trained_defect_models/trained_models/' structure.
                # So if S3_MODEL_PREFIX is 'pre_trained_defect_models/', we arguably should append 'trained_models/' 
                # if the local file is in a 'trained_models' dir.
                
                # To be fail-safe: just upload to S3_MODEL_PREFIX + 'trained_models/'
                # OR simpler: just S3_MODEL_PREFIX 
                
                # Correct Logic: 
                # If S3_MODEL_PREFIX is "pre_trained_defect_models/", we want to put the model in a subfolder "trained_models" 
                # IF that corresponds to how we organize things.
                # Let's just append "trained_models" to prompt user convention.
                
                target_prefix = s3_prefix.rstrip('/') + "/trained_models"
                
                # Generate versioned filename for S3
                # e.g., mymodel.joblib -> mymodel_v20231027103015.joblib
                # Note: 'datetime' is already imported at top level
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                name_stem = Path(filename).stem
                versioned_filename = f"{name_stem}_v{timestamp}.joblib"
                
                logger.info(f"🚀 Detected S3_BUCKET. Uploading to: s3://{s3_bucket}/{target_prefix}/{versioned_filename}...")
                
                upload_success = upload_file_to_s3(
                    local_path=save_path,
                    bucket_name=s3_bucket,
                    s3_prefix=target_prefix,
                    object_name=versioned_filename,
                    aws_region=os.getenv('AWS_REGION', 'us-east-1')
                )
                
                if upload_success:
                    logger.info(f"✓ Successfully uploaded {versioned_filename} to S3")
                else:
                    logger.warning(f"⚠ Failed to upload {versioned_filename} to S3")

            return save_path
            
        except Exception as e:
            logger.error(f"Failed to save model to {save_path}: {e}")
            raise RuntimeError(f"Failed to save model: {e}")

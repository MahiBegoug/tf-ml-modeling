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
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to save model to {save_path}: {e}")
            raise RuntimeError(f"Failed to save model: {e}")

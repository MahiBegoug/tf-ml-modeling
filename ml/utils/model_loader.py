"""
Model Loader Utility
===================

Handles loading, saving, and path resolution for ML models.
"""

import os
import joblib
from typing import Any, Dict, Optional, Tuple, Union
from ml.utils.logger import logger

class ModelLoader:
    """
    Handles model file operations and loading.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir

    def resolve_model_path(self, model_name_or_path: str) -> str:
        """
        Resolve model name to full path.
        
        Args:
            model_name_or_path: Model name or path
            
        Returns:
            Full path to model file
        """
        # If it's already a path and exists, use it
        if os.path.exists(model_name_or_path):
            return model_name_or_path
        
        # If it ends with .joblib, treat as filename
        if model_name_or_path.endswith('.joblib'):
            path = os.path.join(self.models_dir, model_name_or_path)
            if os.path.exists(path):
                return path
        
        # Try with _model.joblib (legacy convention)
        path = os.path.join(self.models_dir, f"{model_name_or_path}_model.joblib")
        if os.path.exists(path):
            return path
            
        # Try with just .joblib (standard convention)
        path = os.path.join(self.models_dir, f"{model_name_or_path}.joblib")
        if os.path.exists(path):
            return path
        
        # Last try: exact name in models_dir
        path = os.path.join(self.models_dir, model_name_or_path)
        if os.path.exists(path):
            return path
        
        raise FileNotFoundError(
            f"Model not found: {model_name_or_path}\n"
            f"Tried:\n"
            f"  - {model_name_or_path}\n"
            f"  - {os.path.join(self.models_dir, model_name_or_path)}\n"
            f"  - {os.path.join(self.models_dir, f'{model_name_or_path}_model.joblib')}"
        )

    def load_model(self, path: str) -> Any:
        """
        Load model using joblib.
        
        Args:
            path: Path to model file
            
        Returns:
            Loaded model data (dict or raw model)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            logger.info(f"Loading model from {path}...")
            model_data = joblib.load(path)
            return model_data
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model from {path}") from e

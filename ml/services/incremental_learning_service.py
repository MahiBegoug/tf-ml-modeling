"""
Incremental Learning Service for Defect Prediction Models

This module provides simple incremental learning capabilities, allowing models
to be updated with new data without full retraining from scratch.

Perfect for initial testing and rapid iteration.
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from ml.utils.metrics import calculate_all_metrics
from ml.utils.model_monitor import ModelMonitor
from ml.utils.logger import logger

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class IncrementalLearningService:
    """
    Service for incremental model updates.
    
    Strategy:
    - Start with a base model
    - Update with new data as it arrives
    - No need to retrain from scratch
    - Fast and simple for initial testing
    """
    
    # Models that support incremental learning
    INCREMENTAL_MODELS = {
        "sgd_classifier": {
            "class": SGDClassifier,
            "params": {
                "loss": "log_loss",  # For probability estimates
                "penalty": "l2",
                "alpha": 0.0001,
                "random_state": 42,
                "max_iter": 1000,
                "class_weight": "balanced"
            }
        },
        "sgd_logistic": {
            "class": SGDClassifier,
            "params": {
                "loss": "log_loss",  # Equivalent to LogisticRegression
                "penalty": "l2",
                "alpha": 0.0001,
                "random_state": 42,
                "max_iter": 1000,
                "class_weight": "balanced"
            }
        },
        "gaussian_nb": {
            "class": GaussianNB,
            "params": {}  # GaussianNB supports partial_fit
        }
    }
    
    # Models that need full retraining (but we can simulate incremental)
    BATCH_MODELS = {
        "randomforest": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": 100,
                "random_state": 42,
                "max_depth": 10,
                "class_weight": "balanced",
                "warm_start": True  # Allows adding more trees
            }
        }
    }
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        BATCH_MODELS["lightgbm"] = {
            "class": lgb.LGBMClassifier,
            "params": {
                "n_estimators": 100,
                "random_state": 42,
                "max_depth": 10,
                "class_weight": "balanced",
                "verbose": -1
            }
        }
    
    def __init__(self, models_dir: str):
        """
        Initialize the incremental learning service.
        
        Args:
            models_dir: Directory to save/load models
        """
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        self.update_history = []
    
    def create_base_model(self, 
                         model_type: str = "sgd_classifier",
                         custom_params: Optional[Dict] = None) -> Any:
        """
        Create a new base model for incremental learning.
        
        Args:
            model_type: Type of model to create
            custom_params: Optional custom parameters
            
        Returns:
            Initialized model instance
        """
        # Check if model supports incremental learning
        if model_type in self.INCREMENTAL_MODELS:
            config = self.INCREMENTAL_MODELS[model_type]
        elif model_type in self.BATCH_MODELS:
            config = self.BATCH_MODELS[model_type]
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported: {list(self.INCREMENTAL_MODELS.keys()) + list(self.BATCH_MODELS.keys())}"
            )
        
        # Get parameters
        params = config["params"].copy()
        if custom_params:
            params.update(custom_params)
        
        # Create model
        model = config["class"](**params)
        
        logger.info(f"Created base model: {model_type}")
        return model
    
    def initial_train(self,
                     feature_vectors: Dict[str, List[float]],
                     labels: Dict[str, int],
                     model_type: str = "sgd_classifier",
                     model_name: str = "incremental_model") -> Dict[str, Any]:
        """
        Initial training of the base model.
        
        Args:
            feature_vectors: Initial training data (block_id -> features)
            labels: Initial labels (block_id -> 0 or 1)
            model_type: Type of model to train
            model_name: Name for saving the model
            
        Returns:
            Dict with training metrics and model info
        """
        logger.info("=" * 70)
        logger.info("INITIAL TRAINING (Incremental Learning)")
        logger.info("=" * 70)
        
        # Align data
        X, y = self._align_data(feature_vectors, labels)
        
        if len(X) == 0:
            raise ValueError("No aligned data for training")
        
        logger.info(f"Training with {len(X)} samples...")
        
        # Create and train model
        model = self.create_base_model(model_type)
        model.fit(X, y)
        
        # Evaluate
        y_pred = model.predict(X)
        metrics = calculate_all_metrics(y, y_pred)
        
        # Add metadata
        metrics["model_type"] = model_type
        metrics["n_samples"] = len(X)
        metrics["n_features"] = X.shape[1]
        metrics["training_type"] = "initial"
        metrics["timestamp"] = datetime.now().isoformat()
        
        # Extract feature names from feature_vectors (use first block's keys as feature order)
        # Since we're using dict, we need to preserve the order
        first_block_id = list(feature_vectors.keys())[0]
        feature_order = list(range(len(feature_vectors[first_block_id])))  # Feature indices
        
        # Save model with metadata (bundle format)
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        
        model_bundle = {
            'model': model,
            'feature_order': feature_order,  # Feature indices in order
            'n_features': X.shape[1],
            'model_type': model_type,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_bundle, model_path)
        metrics["model_path"] = model_path
        
        # Save metadata (ModelMonitor)
        monitor = ModelMonitor(self.models_dir)
        monitor.log_training(model_name, metrics, append=False)
        
        logger.info(f"✅ Initial training complete")
        logger.info(f"   MCC: {metrics['mcc']:.3f}, G-Mean: {metrics['gmean']:.3f}")
        logger.info(f"   Model saved to: {model_path}")
        logger.info(f"   Features: {X.shape[1]} (order preserved)")
        
        return metrics
    
    def incremental_update(self,
                          model_name: str,
                          new_features: Dict[str, List[float]],
                          new_labels: Dict[str, int]) -> Dict[str, Any]:
        """
        Update existing model with new data (incremental learning).
        
        Args:
            model_name: Name of the model to update
            new_features: New feature vectors
            new_labels: New labels
            
        Returns:
            Dict with update metrics
        """
        logger.info("=" * 70)
        logger.info("INCREMENTAL UPDATE")
        logger.info("=" * 70)
        
        # Load existing model bundle
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model_bundle = joblib.load(model_path)
        
        # Handle both bundle format and legacy format
        if isinstance(model_bundle, dict):
            model = model_bundle.get('model', model_bundle) # Handle if 'model' key missing but dict
            feature_order = model_bundle.get('feature_order', None)
            n_features = model_bundle.get('n_features', None)
            scaler = model_bundle.get('scaler', None)
            
            # Legacy check: if model is the dict but has no 'model' key (unlikely with new format)
            if not isinstance(model, (list, tuple, np.ndarray)) and not hasattr(model, 'predict') and 'model' not in model_bundle:
                 # This might happen if the bundle IS the model object but acts like a dict? No.
                 pass

            logger.info(f"Loaded model bundle from: {model_path}")
            logger.info(f"  Features: {n_features}, Scaler: {'✓' if scaler else '✗'}")
        else:
            # Legacy format (just the model)
            model = model_bundle
            feature_order = None
            n_features = None
            scaler = None
            logger.info(f"Loaded model (legacy format) from: {model_path}")
        
        # Align new data
        X_new, y_new = self._align_data(new_features, new_labels)
        
        if len(X_new) == 0:
            logger.warning("No new data to update with")
            return {"status": "skipped", "reason": "no_data"}
        
        # Verify feature count matches
        if n_features is not None and X_new.shape[1] != n_features:
            raise ValueError(
                f"Feature count mismatch! Model expects {n_features} features, "
                f"but got {X_new.shape[1]} features. Ensure feature order is preserved."
            )
        
        logger.info(f"Updating with {len(X_new)} new samples...")

        # Update Scaler if present
        if scaler:
            logger.info("Updating scaler...")
            if hasattr(scaler, 'partial_fit'):
                scaler.partial_fit(X_new)
            X_new = scaler.transform(X_new)
        
        # Check if model supports incremental learning
        if hasattr(model, 'partial_fit'):
            # True incremental learning (SGD, etc.)
            model.partial_fit(X_new, y_new, classes=[0, 1])
            update_method = "partial_fit"
        elif LIGHTGBM_AVAILABLE and isinstance(model, lgb.LGBMClassifier):
            # LightGBM training continuation
            model.fit(X_new, y_new, init_model=model.booster_)
            update_method = "training_continuation"
        elif hasattr(model, 'warm_start') and model.warm_start:
            # Warm start (RandomForest with warm_start=True)
            model.n_estimators += 50  # Add 50 more trees
            model.fit(X_new, y_new)
            update_method = "warm_start"
        else:
            # Fallback
            logger.warning("Model doesn't support incremental learning")
            return {"status": "error", "reason": "incremental_not_supported"}
        
        # Evaluate on new data
        y_pred = model.predict(X_new)
        metrics = calculate_all_metrics(y_new, y_pred)
        
        # Add metadata
        metrics["update_method"] = update_method
        metrics["n_new_samples"] = len(X_new)
        metrics["n_features"] = X_new.shape[1]
        metrics["timestamp"] = datetime.now().isoformat()
        
        # Save updated model bundle
        updated_bundle = {
            'model': model,
            'scaler': scaler,
            'feature_order': feature_order if feature_order is not None else list(range(X_new.shape[1])),
            'n_features': X_new.shape[1],
            'model_type': model_bundle.get('model_type', 'unknown') if isinstance(model_bundle, dict) else 'unknown',
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(updated_bundle, model_path)
        metrics["model_path"] = model_path
        
        # Log to monitor (Legacy)
        monitor = ModelMonitor(self.models_dir)
        monitor.log_training(model_name, metrics, append=True)

        # MLflow Logging
        import mlflow
        mlflow.set_experiment("IncrementalLearning")
        
        with mlflow.start_run():
            mlflow.log_param("update_method", update_method)
            mlflow.log_param("n_new_samples", len(X_new))
            
            # Log Metrics
            safe_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(safe_metrics)
            
            # Log Model Bundle as Artifact (to preserve state)
            mlflow.log_artifact(model_path)
        
        logger.info(f"✅ Incremental update complete")
        logger.info(f"   Method: {update_method}")
        logger.info(f"   MCC: {metrics['mcc']:.3f}, G-Mean: {metrics['gmean']:.3f}")
        
        return metrics
    
    def _align_data(self, 
                    feature_vectors: Dict[str, List[float]], 
                    labels: Dict[str, int]) -> tuple:
        """
        Align features and labels by block_id.
        
        Returns:
            (X, y) as numpy arrays
        """
        X = []
        y = []
        
        for block_id, vector in feature_vectors.items():
            if block_id in labels:
                X.append(vector)
                y.append(labels[block_id])
        
        return np.array(X), np.array(y)
    
    # Removed _save_metadata, delegating to ModelMonitor
    

import os
import joblib
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from ml.utils.metrics import calculate_all_metrics
from ml.utils.model_monitor import ModelMonitor
from ml.utils.logger import logger

class RetrainingService:
    """
    Service responsible for retraining defect prediction models.
    Supports multiple model types: RandomForest, LightGBM, LogisticRegression, NaiveBayes, etc.
    """

    # Model registry with default configurations
    MODEL_REGISTRY = {
        "randomforest": {
            "class": RandomForestClassifier,
            "params": {"n_estimators": 100, "random_state": 42, "max_depth": 10}
        },
        "logisticreg": {
            "class": LogisticRegression,
            "params": {"random_state": 42, "max_iter": 1000}
        },
        "naivebayes": {
            "class": GaussianNB,
            "params": {}
        },
        "gradientboosting": {
            "class": GradientBoostingClassifier,
            "params": {"n_estimators": 100, "random_state": 42, "learning_rate": 0.1}
        },
        "decisiontree": {
            "class": DecisionTreeClassifier,
            "params": {"random_state": 42, "max_depth": 10}
        }
    }

    def __init__(self, models_dir: str):
        """
        Args:
            models_dir (str): Directory where models are saved.
        """
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        # Initialize ModelSaver
        from ml.utils.model_saver import ModelSaver
        self.saver = ModelSaver(models_dir)

    def _get_model_instance(self, model_type: str, custom_params: Optional[Dict] = None):
        """
        Creates a model instance based on the specified type.
        
        Args:
            model_type: Type of model (e.g., 'randomforest', 'logisticreg')
            custom_params: Optional custom parameters to override defaults
            
        Returns:
            Instantiated model object
        """
        model_type = model_type.lower()
        
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(self.MODEL_REGISTRY.keys())}"
            )
        
        model_config = self.MODEL_REGISTRY[model_type]
        model_class = model_config["class"]
        params = model_config["params"].copy()
        
        # Override with custom parameters if provided
        if custom_params:
            params.update(custom_params)
        
        logger.info(f"Creating {model_type} model with params: {params}")
        return model_class(**params)

    def train(self, 
              feature_vectors: Dict[str, List[float]], 
              labels: Dict[str, int], 
              model_type: str = "randomforest",
              model_name: Optional[str] = None,
              custom_params: Optional[Dict] = None,
              save: bool = True) -> Dict[str, float]:
        """
        Trains a model using the provided data.

        Args:
            feature_vectors: Map of block_id -> feature vector.
            labels: Map of block_id -> 0 or 1 (defect).
            model_type: Type of model to train (e.g., 'randomforest', 'logisticreg', 'naivebayes').
            model_name: Name of the output model file (without extension). 
                       If None, uses model_type as the name.
            custom_params: Optional custom hyperparameters for the model.
            save: Whether to save the model to disk.

        Returns:
            Dict[str, float]: Training metrics (Accuracy, F1, etc.)
        """
        if not feature_vectors or not labels:
            logger.warning("No data provided for training.")
            return {}

        # Use model_type as default name if not provided
        if model_name is None:
            model_name = f"{model_type}_model"

        # Align data
        X = []
        y = []
        
        aligned_count = 0
        for block_id, vector in feature_vectors.items():
            if block_id in labels:
                X.append(vector)
                y.append(labels[block_id])
                aligned_count += 1
        
        if aligned_count == 0:
            raise ValueError("No matching block_ids found between features and labels.")

        logger.info(f"Training {model_type} model with {aligned_count} samples...")

        # Create model instance
        clf = self._get_model_instance(model_type, custom_params)
        
        # Train
        clf.fit(X, y)
        
        # Evaluate on Training Data
        y_pred = clf.predict(X)
        
        # Calculate all metrics using the metrics module
        metrics = calculate_all_metrics(y, y_pred)
        
        # Add metadata
        metrics["model_type"] = model_type
        metrics["n_samples"] = aligned_count
        
        logger.info(f"Training completed. Metrics: {metrics}")

        # Save
        if save:
            try:
                # Use ModelSaver to create a compatible bundle
                saved_path = self.saver.save_model(
                    model=clf,
                    model_name=model_name,
                    scaler=None,  # Base RetrainingService doesn't use scaler
                    metadata={
                        "model_type": model_type,
                        "metrics": metrics,
                        "n_samples": aligned_count
                    }
                )
                metrics["model_path"] = saved_path
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
                raise
            
            # Log training metrics
            try:
                monitor = ModelMonitor(self.models_dir)
                metrics["training_type"] = "full_retrain"
                metrics["timestamp"] = joblib.os.path.getmtime(saved_path) if os.path.exists(saved_path) else joblib.os.time.time()
                # Format timestamp nicely
                import datetime
                metrics["timestamp"] = datetime.datetime.fromtimestamp(metrics["timestamp"]).isoformat()
                
                monitor.log_training(model_name, metrics, append=True)
            except Exception as e:
                logger.warning(f"Failed to log training history: {e}")

        return metrics

    def train_multiple(self,
                      feature_vectors: Dict[str, List[float]],
                      labels: Dict[str, int],
                      model_types: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Train multiple model types and compare their performance.
        
        Args:
            feature_vectors: Map of block_id -> feature vector.
            labels: Map of block_id -> 0 or 1 (defect).
            model_types: List of model types to train. If None, trains all available.
            
        Returns:
            Dict mapping model_type to its metrics
        """
        if model_types is None:
            model_types = list(self.MODEL_REGISTRY.keys())
        
        results = {}
        logger.info(f"Training {len(model_types)} models: {model_types}")
        
        for model_type in model_types:
            try:
                metrics = self.train(
                    feature_vectors=feature_vectors,
                    labels=labels,
                    model_type=model_type,
                    save=True
                )
                results[model_type] = metrics
                logger.info(
                    f"✓ {model_type}: MCC={metrics['mcc']:.3f}, G-Mean={metrics['gmean']:.3f}, "
                    f"F1={metrics['f1']:.3f}, Accuracy={metrics['accuracy']:.3f}"
                )
            except Exception as e:
                logger.error(f"✗ Failed to train {model_type}: {e}")
                results[model_type] = {"error": str(e)}
        
        return results


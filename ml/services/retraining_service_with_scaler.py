"""
Enhanced Retraining Service with MinMaxScaler Support

This version automatically includes MinMaxScaler to match your existing model format.
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

from ml.utils.metrics import calculate_all_metrics
from ml.utils.model_monitor import ModelMonitor
from ml.utils.logger import logger


class RetrainingServiceWithScaler:
    """
    Enhanced retraining service that includes MinMaxScaler.
    
    Saves models in the same format as your existing models:
    {
        'model': <trained model>,
        'scaler': <fitted MinMaxScaler>
    }
    """
    MODEL_REGISTRY = {
        "randomforest": {
            "class": RandomForestClassifier,
            "params": {"n_estimators": 100, "random_state": 42, "max_depth": 10, "class_weight": "balanced"}
        },
        "extratrees": {
            "class": ExtraTreesClassifier,
            "params": {"n_estimators": 100, "random_state": 42, "max_depth": 10, "class_weight": "balanced"}
        },
        "lightgbm": {
            "class": LGBMClassifier,
            "params": {"n_estimators": 100, "random_state": 42, "max_depth": -1, "class_weight": "balanced"}
        },
        "logisticreg": {
            "class": LogisticRegression,
            "params": {"random_state": 42, "max_iter": 1000, "class_weight": "balanced"}
        },
        "naivebayes": {
            "class": GaussianNB,
            "params": {}
        },
        "decisiontree": {
            "class": DecisionTreeClassifier,
            "params": {"random_state": 42, "max_depth": 10, "class_weight": "balanced"}
        },
        "dummy": {
            "class": DummyClassifier,
            "params": {"strategy": "stratified", "random_state": 42}
        },
        "svm": {
            "class": SVC,
            "params": {"random_state": 42, "probability": True, "class_weight": "balanced"}
        },
        "mlp": {
            "class": MLPClassifier,
            "params": {"random_state": 42, "max_iter": 500, "early_stopping": True}
        }
    }
    
    def __init__(self, models_dir: str, use_scaler: bool = True):
        """
        Initialize the retraining service.
        
        Args:
            models_dir: Directory to save models
            use_scaler: Whether to use MinMaxScaler (default: True)
        """
        self.models_dir = models_dir
        self.use_scaler = use_scaler
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    
    def train(self,
              feature_vectors: Dict[str, List[float]],
              labels: Dict[str, int],
              model_type: str = "randomforest",
              custom_params: Optional[Dict] = None,
              model_name: str = "model") -> Dict[str, Any]:
        """
        Train a model with MinMaxScaler.
        
        Args:
            feature_vectors: Feature vectors (block_id -> features)
            labels: Labels (block_id -> 0 or 1)
            model_type: Type of model to train
            custom_params: Optional custom parameters
            model_name: Name for saving the model
            
        Returns:
            Dict with metrics and model path
        """
        logger.info("=" * 70)
        logger.info(f"TRAINING: {model_type.upper()} (with MinMaxScaler)")
        logger.info("=" * 70)
        
        # Align data
        X, y = self._align_data(feature_vectors, labels)
        
        if len(X) == 0:
            raise ValueError("No aligned data for training")
        
        logger.info(f"Training samples: {len(X)}")
        logger.info(f"Features: {X.shape[1]}")
        logger.info(f"Defective: {sum(y)}, Non-defective: {len(y) - sum(y)}")
        
        # Fit scaler
        scaler = None
        X_scaled = X
        
        if self.use_scaler:
            logger.info("Fitting MinMaxScaler...")
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
        
        # Create and train model
        model = self._get_model_instance(model_type, custom_params)
        logger.info(f"Training {model_type}...")
        model.fit(X_scaled, y)
        
        # Evaluate
        y_pred = model.predict(X_scaled)
        metrics = calculate_all_metrics(y, y_pred)
        
        # Add metadata
        metrics["model_type"] = model_type
        metrics["n_samples"] = len(X)
        metrics["n_features"] = X.shape[1]
        metrics["uses_scaler"] = self.use_scaler
        metrics["timestamp"] = datetime.now().isoformat()
        
        # Save model (with scaler if used)
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        
        if self.use_scaler:
            model_data = {
                'model': model,
                'scaler': scaler
            }
            joblib.dump(model_data, model_path)
            logger.info(f"Saved model with scaler to: {model_path}")
        else:
            joblib.dump(model, model_path)
            logger.info(f"Saved model to: {model_path}")
        
        metrics["model_path"] = model_path
        
        # Log training metrics (ModelMonitor)
        try:
            monitor = ModelMonitor(self.models_dir)
            metrics["training_type"] = "full_retrain_with_scaler"
            metrics["timestamp"] = datetime.now().isoformat()
            monitor.log_training(model_name, metrics, append=True)
        except Exception as e:
            logger.warning(f"Failed to log training history: {e}")
        
        logger.info(f"\n✓ Training complete")
        logger.info(f"  MCC: {metrics['mcc']:.3f}, G-Mean: {metrics['gmean']:.3f}")
        
        return metrics
    
    def _get_model_instance(self, model_type: str, custom_params: Optional[Dict] = None):
        """Get model instance with parameters."""
        model_type = model_type.lower()
        
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(self.MODEL_REGISTRY.keys())}"
            )
        
        config = self.MODEL_REGISTRY[model_type]
        model_class = config["class"]
        params = config["params"].copy()
        
        if custom_params:
            params.update(custom_params)
        
        return model_class(**params)
    
    def _align_data(self, feature_vectors: Dict, labels: Dict) -> tuple:
        """Align features and labels."""
        X = []
        y = []
        
        for block_id, vector in feature_vectors.items():
            if block_id in labels:
                X.append(vector)
                y.append(labels[block_id])
        
        return np.array(X), np.array(y)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Train model with MinMaxScaler (matching your existing format)
    """
    import sys
    sys.path.insert(0, 'c:\\Users\\Admin\\PycharmProjects\\TFDefectGA')
    import tempfile
    
    # Create sample data
    np.random.seed(42)
    features = {
        f"block{i}": np.random.rand(56).tolist()  # 56 features like randomforest
        for i in range(100)
    }
    labels = {
        f"block{i}": 0 if i < 80 else 1
        for i in range(100)
    }
    
    # Train with scaler
    with tempfile.TemporaryDirectory() as tmpdir:
        service = RetrainingServiceWithScaler(tmpdir, use_scaler=True)
        
        metrics = service.train(
            feature_vectors=features,
            labels=labels,
            model_type="randomforest",
            model_name="test_model"
        )
        
        print(f"\n✓ Model trained with scaler")
        print(f"  MCC: {metrics['mcc']:.3f}")
        print(f"  Model saved to: {metrics['model_path']}")
        
        # Verify format
        model_data = joblib.load(metrics['model_path'])
        print(f"\n✓ Model format verification:")
        print(f"  Type: {type(model_data)}")
        print(f"  Keys: {model_data.keys() if isinstance(model_data, dict) else 'Not a dict'}")
        
        if isinstance(model_data, dict):
            print(f"  Model type: {type(model_data['model']).__name__}")
            print(f"  Scaler type: {type(model_data['scaler']).__name__}")
            
            # Test prediction with scaler
            X_test = np.random.rand(10, 56)
            X_scaled = model_data['scaler'].transform(X_test)
            predictions = model_data['model'].predict(X_scaled)
            print(f"\n✓ Test prediction successful")
            print(f"  Predictions: {predictions}")

"""
Enhanced Prediction Service with Scaler Support

This service properly handles:
1. Loading model + scaler from .joblib files
2. Automatic scaling before prediction
3. Model selection by name or path
4. Probability predictions for individual blocks
5. Single instance prediction (for specific block changes)
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from ml.utils.logger import logger


class PredictionService:
    """
    Enhanced prediction service with scaler support and model selection.
    
    Features:
    - Loads model + scaler automatically
    - Applies scaler before prediction
    - Supports model selection by name
    - Provides probability predictions
    - Can predict on single instances
    """

    def __init__(self, model_name_or_path: str, models_dir: str = "models", threshold: float = 0.5):
        """
        Initialize the service using helper classes for loading and validation.
        """
        from ml.utils.model_loader import ModelLoader
        from ml.utils.feature_validator import FeatureValidator

        self.models_dir = models_dir
        self.threshold = threshold
        
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        # Initialize helpers
        self.loader = ModelLoader(models_dir)
        
        # Load model data
        self.model_path = self.loader.resolve_model_path(model_name_or_path)
        model_data = self.loader.load_model(self.model_path)
        
        # Parse model data
        self.scaler = None
        self.feature_order = None
        self.n_features = None
        self.metadata = {}
        
        if isinstance(model_data, dict):
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.metadata = model_data.get('metadata', {})
            self.feature_order = model_data.get('feature_order', None)
            self.n_features = model_data.get('n_features', None)
            
            if self.model is None:
                raise ValueError(f"Model dict missing 'model' key: {model_data.keys()}")
            
            if self.feature_order:
                logger.info(f"✓ Model loaded with feature order ({self.n_features} features)")
        else:
            self.model = model_data
            logger.warning("Model loaded without scaler/metadata.")

        # Fallback: Try to infer n_features if not provided in metadata
        if self.n_features is None:
            if hasattr(self.model, "n_features_in_"):
                self.n_features = self.model.n_features_in_
            elif self.scaler and hasattr(self.scaler, "n_features_in_"):
                self.n_features = self.scaler.n_features_in_

        # Initialize validator
        self.validator = FeatureValidator(self.n_features, self.scaler)
        
        # Expose scaler from validator (if created internally)
        self.scaler = self.validator.scaler
        
        logger.info(f"✓ Service ready: {type(self.model).__name__} | Threshold: {self.threshold:.0%}")

    # Delegated methods
    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        return self.validator.apply_scaler(X)
    
    def _validate_features(self, X: np.ndarray) -> None:
        self.validator.validate_features(X)
        
    # Helper for backward compatibility or internal use
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "model_type": type(self.model).__name__,
            "model_path": self.model_path,
            "threshold": self.threshold,
            "has_scaler": self.scaler is not None,
            "scaler_type": type(self.scaler).__name__ if self.scaler else None,
            "supports_proba": hasattr(self.model, "predict_proba"),
            "metadata": self.metadata
        }
        if self.n_features:
            info["n_features"] = self.n_features
        return info

    def predict(self, feature_vectors: Dict[str, List[float]]) -> Dict[str, int]:
        """
        Predict defects for a batch of blocks.

        Args:
            feature_vectors: Dictionary mapping block_id to RAW feature list
                            (scaler will be applied automatically)

        Returns:
            Dictionary mapping block_id to prediction (0 or 1)
        
        Example:
            predictions = service.predict({
                "block1": [0.5, 0.3, ...],  # Raw features
                "block2": [0.2, 0.8, ...]
            })
            # → {'block1': 0, 'block2': 1}
        """
        if not feature_vectors:
            return {}

        block_ids = list(feature_vectors.keys())
        X = np.array(list(feature_vectors.values()))
        
        try:
            # Validate feature count and alignment
            self._validate_features(X)
            
            # Apply scaler if available
            X_scaled = self._apply_scaler(X)
            
            # Respect threshold if model supports probabilities
            if hasattr(self.model, "predict_proba"):
                # Use manual thresholding on probabilities
                probs = self.model.predict_proba(X_scaled)
                logger.info(f"DEBUG: probs type: {type(probs)}, shape: {getattr(probs, 'shape', 'NoShape')}, val: {probs}")
                
                # Handle binary classification (take class 1 prob)
                # Handle binary classification (take class 1 prob)
                if len(probs.shape) >= 2 and probs.shape[1] == 2:
                    probs_class1 = probs[:, 1]
                elif len(probs.shape) >= 2:
                    probs_class1 = probs[:, 0] # Fallback
                else:
                    # 1D array case
                    probs_class1 = probs
                
                predictions = (probs_class1 >= self.threshold).astype(int)
            else:
                # Fallback to model's default threshold
                predictions = self.model.predict(X_scaled)
            
            # Map back to block_ids
            return {
                block_id: int(pred) 
                for block_id, pred in zip(block_ids, predictions)
            }
            
        except Exception as e:
            logger.exception(f"Prediction failed: {e}")
            raise RuntimeError("Model prediction failed") from e

    def predict_proba(self, feature_vectors: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Predict defect probabilities for a batch of blocks.

        Args:
            feature_vectors: Dictionary mapping block_id to RAW feature list

        Returns:
            Dictionary mapping block_id to probability of defect (class 1)
        
        Example:
            probabilities = service.predict_proba({
                "block1": [0.5, 0.3, ...],
                "block2": [0.2, 0.8, ...]
            })
            # → {'block1': 0.15, 'block2': 0.92}
        """
        if not feature_vectors:
            return {}

        block_ids = list(feature_vectors.keys())
        X = np.array(list(feature_vectors.values()))

        try:
            # Check if model supports predict_proba
            if not hasattr(self.model, "predict_proba"):
                logger.warning("Model does not support predict_proba. Returning hard predictions.")
                preds = self.predict(feature_vectors)
                return {k: float(v) for k, v in preds.items()}

            # Validate feature count and alignment
            self._validate_features(X)
            
            # Apply scaler if available
            X_scaled = self._apply_scaler(X)
            
            # Get probabilities
            probs = self.model.predict_proba(X_scaled)
            
            # Extract probability of class 1 (defect)
            result = {}
            for i, block_id in enumerate(block_ids):
                if len(probs.shape) >= 2 and probs.shape[1] == 2:
                    # Binary classification: take probability of class 1
                    result[block_id] = float(probs[i][1])
                elif len(probs.shape) >= 2:
                    # Fallback
                    result[block_id] = float(probs[i][0])
                else:
                     # 1D array
                    result[block_id] = float(probs[i])

            return result

        except Exception as e:
            logger.error(f"Prediction probability failed: {e}")
            raise RuntimeError("Probability prediction failed") from e

    def predict_single(self, 
                       block_id: str, 
                       features: List[float]) -> Tuple[int, float]:
        """
        Predict defect for a SINGLE block (useful for testing specific instances).

        Args:
            block_id: Block identifier
            features: RAW feature vector (scaler will be applied)

        Returns:
            Tuple of (prediction, probability)
            - prediction: 0 (clean) or 1 (defective)
            - probability: confidence score (0.0 to 1.0)
        
        Example:
            pred, prob = service.predict_single(
                "resource.aws_instance.web",
                [0.5, 0.3, 0.8, ...]
            )
            # → (1, 0.92)  # Defective with 92% confidence
        """
        # Use batch methods with single item
        predictions = self.predict({block_id: features})
        probabilities = self.predict_proba({block_id: features})
        
        return predictions[block_id], probabilities[block_id]

    def predict_with_confidence(self, 
                                feature_vectors: Dict[str, List[float]]) -> Dict[str, Tuple[int, float]]:
        """
        Predict defects with confidence scores for all blocks.

        Args:
            feature_vectors: Dictionary mapping block_id to RAW feature list

        Returns:
            Dictionary mapping block_id to (prediction, probability)
        
        Example:
            results = service.predict_with_confidence({
                "block1": [0.5, 0.3, ...],
                "block2": [0.2, 0.8, ...]
            })
            # → {
            #     'block1': (0, 0.15),  # Clean with 15% defect probability
            #     'block2': (1, 0.92)   # Defective with 92% defect probability
            # }
        """
        predictions = self.predict(feature_vectors)
        probabilities = self.predict_proba(feature_vectors)
        
        return {
            block_id: (predictions[block_id], probabilities[block_id])
            for block_id in predictions.keys()
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_type": type(self.model).__name__,
            "model_path": self.model_path,
            "threshold": self.threshold,
            "has_scaler": self.scaler is not None,
            "scaler_type": type(self.scaler).__name__ if self.scaler else None,
            "supports_proba": hasattr(self.model, "predict_proba"),
            "metadata": self.metadata
        }
        
        # Add feature count if available
        if hasattr(self.model, "n_features_in_"):
            info["n_features"] = self.model.n_features_in_
        elif self.scaler and hasattr(self.scaler, "n_features_in_"):
            info["n_features"] = self.scaler.n_features_in_
        
        return info

    def describe(self) -> str:
        """
        Get a human-readable description of the model.

        Returns:
            Description string
        """
        info = self.get_model_info()
        
        desc = f"🧠 Model: {info['model_type']}"
        
        if info['has_scaler']:
            desc += f" | 🔧 Scaler: {info['scaler_type']}"
        
        if 'n_features' in info:
            desc += f" | 🔁 Features: {info['n_features']}"
        
        desc += f" | 🎯 Threshold: {self.threshold:.0%}"
        
        if info['supports_proba']:
            desc += " | 📊 Supports probabilities"
        
        return desc

    def predict_with_threshold(self, 
                               feature_vectors: Dict[str, List[float]], 
                               threshold: Optional[float] = None) -> Dict[str, int]:
        """
        Predict defects using a custom threshold (overrides default).

        Args:
            feature_vectors: Dictionary mapping block_id to RAW feature list
            threshold: Custom threshold (if None, uses self.threshold)

        Returns:
            Dictionary mapping block_id to prediction (0 or 1)
        
        Example:
            # Use default threshold (50%)
            predictions = service.predict_with_threshold(features)
            
            # Use conservative threshold (30%)
            predictions = service.predict_with_threshold(features, threshold=0.3)
            
            # Use strict threshold (70%)
            predictions = service.predict_with_threshold(features, threshold=0.7)
        """
        if threshold is None:
            threshold = self.threshold
        
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        # Get probabilities
        probabilities = self.predict_proba(feature_vectors)
        
        # Apply threshold
        return {
            block_id: 1 if prob >= threshold else 0
            for block_id, prob in probabilities.items()
        }

    def set_threshold(self, threshold: float) -> None:
        """
        Update the decision threshold.

        Args:
            threshold: New threshold value (0.0 to 1.0)
        
        Example:
            service = PredictionService("lightgbm")  # Default 0.5
            service.set_threshold(0.3)  # Now use 30% threshold
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        old_threshold = self.threshold
        self.threshold = threshold
        logger.info(f"Threshold updated: {old_threshold:.2%} → {threshold:.2%}")

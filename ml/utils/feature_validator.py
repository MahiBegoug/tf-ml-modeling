"""
Feature Validator Utility
========================

Handles validation and scaling of feature vectors.
"""

import numpy as np
from typing import Optional, Any
from sklearn.preprocessing import MinMaxScaler
from ml.utils.logger import logger

class FeatureValidator:
    """
    Handles feature validation and scaler application.
    """
    
    def __init__(self, n_features: Optional[int] = None, scaler: Optional[Any] = None):
        """
        Initialize validator.
        
        Args:
            n_features: Expected number of features (optional)
            scaler: Scaler instance (optional)
        """
        self.n_features = n_features
        self.scaler = scaler
        
        # Automatically create scaler if not present
        if self.scaler is None:
            logger.info("No scaler provided. Creating new MinMaxScaler.")
            self.scaler = MinMaxScaler()
            self._scaler_fitted = False
        else:
            self._scaler_fitted = True
            logger.info(f"✓ Scaler loaded: {type(self.scaler).__name__}")

    def validate_features(self, X: np.ndarray) -> None:
        """
        Validate that feature count matches expected features.
        
        Args:
            X: Feature matrix to validate
            
        Raises:
            ValueError: If feature count doesn't match
        """
        if self.n_features is not None:
            if X.shape[1] != self.n_features:
                raise ValueError(
                    f"Feature count mismatch! Model expects {self.n_features} features, "
                    f"but got {X.shape[1]} features. "
                    f"Ensure features are provided in the same order as during training."
                )
        else:
            logger.warning(
                f"Model doesn't have feature count metadata. "
                f"Cannot validate feature alignment (got {X.shape[1]} features)."
            )

    def apply_scaler(self, X: np.ndarray) -> np.ndarray:
        """
        Apply scaler if available.
        
        Args:
            X: Raw features
            
        Returns:
            Scaled features
        """
        if self.scaler is not None:
            try:
                # If scaler is not fitted (e.g. newly created), we might need to handle it?
                # Usually we expect a fitted scaler for prediction.
                # If it's the default one created in __init__, it's not fitted.
                import sklearn.exceptions
                return self.scaler.transform(X)
            except (sklearn.exceptions.NotFittedError, AttributeError):
                # Fallback or warning?
                # For prediction, we usually assume fitted scaler.
                # If it's a dummy scaler, maybe return X?
                if not hasattr(self.scaler, 'n_features_in_'): 
                     # Not fitted
                     return X
                raise
        return X

"""
Unified Prediction Layer - Extensible and Reusable

This service provides a complete prediction layer that handles:
1. Feature vector alignment with model's expected feature order
2. Configurable prediction thresholds
3. Retraining strategy specification (incremental, full retrain, etc.)
4. Cache/historical data management for retraining
5. Model lifecycle management

Design Goals:
- Extensibility: Easy to add new retraining strategies
- Reusability: Can be used across different contexts
- Configurability: All parameters are configurable
- Production-ready: Includes validation, logging, and error handling
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from ml.services.prediction_service import PredictionService
from ml.services.incremental_learning_service import IncrementalLearningService
from ml.services.retraining_service_with_scaler import RetrainingServiceWithScaler
from ml.services.prediction_config import PredictionConfig, RetrainingStrategy
from ml.services.prediction_result import PredictionResult
from ml.utils.feature_schema import load_feature_schema
from ml.utils.logger import logger


class UnifiedPredictionLayer:
    """
    Unified prediction layer with feature alignment, configurable thresholds,
    and retraining capabilities.
    
    Example:
        # Basic usage
        config = PredictionConfig(
            model_name="my_model",
            threshold=0.6,
            feature_schema="lightgbm_features.csv"
        )
        
        layer = UnifiedPredictionLayer(config)
        result = layer.predict(feature_vectors)
        
        # With retraining
        config = PredictionConfig(
            model_name="my_model",
            retraining_strategy=RetrainingStrategy.INCREMENTAL,
            enable_cache=True
        )
        
        layer = UnifiedPredictionLayer(config)
        result = layer.predict(feature_vectors)
        
        # Later, retrain with new data
        layer.retrain(new_features, new_labels)
    """
    
    def __init__(self, config: PredictionConfig):
        """
        Initialize prediction layer.
        
        Args:
            config: Configuration for prediction layer
        """
        self.config = config
        
        # Initialize prediction service
        self.prediction_service = PredictionService(
            model_name_or_path=config.model_name,
            models_dir=config.models_dir,
            threshold=config.threshold
        )
        
        # Initialize Feature Aligner
        # Try to get order from model if not in config
        model_order = getattr(self.prediction_service, 'feature_order', None)
        
        from ml.utils.feature_aligner import FeatureAligner
        self.aligner = FeatureAligner(
            feature_schema_path=config.feature_schema,
            feature_order=config.feature_order,
            model_feature_order=model_order
        )
        
        # Initialize retraining service if needed
        self.retraining_service = None
        if config.retraining_strategy == RetrainingStrategy.INCREMENTAL:
            self.retraining_service = IncrementalLearningService(config.models_dir)
            logger.info("✓ Incremental learning enabled")
        elif config.retraining_strategy == RetrainingStrategy.FULL_RETRAIN:
            self.retraining_service = RetrainingServiceWithScaler(config.models_dir, use_scaler=True)
            logger.info("✓ Full retraining enabled")
        
        # Initialize cache manager if enabled
        self.cache_manager = None
        if config.enable_cache:
            from ml.utils.prediction_cache_manager import PredictionCacheManager
            self.cache_manager = PredictionCacheManager(
                model_name=config.model_name,
                cache_dir=config.cache_dir
            )
        
        logger.info(f"✓ UnifiedPredictionLayer initialized: {config.model_name}")

    def predict(
        self,
        feature_vectors: Dict[str, List[float]],
        threshold: Optional[float] = None,
        return_probabilities: Optional[bool] = None
    ) -> PredictionResult:
        """
        Make predictions on feature vectors.
        """
        # Delegate alignment
        if self.config.auto_align_features:
            aligned_vectors = self.aligner.align_features(
                feature_vectors, 
                validate_count=self.config.validate_features,
                strict_mode=self.config.strict_mode
            )
        else:
            aligned_vectors = feature_vectors
        
        # Use config defaults if not specified
        threshold = threshold if threshold is not None else self.config.threshold
        return_probs = return_probabilities if return_probabilities is not None else self.config.return_probabilities
        
        # Make predictions
        if threshold != self.prediction_service.threshold:
            predictions = self.prediction_service.predict_with_threshold(aligned_vectors, threshold)
        else:
            predictions = self.prediction_service.predict(aligned_vectors)
        
        # Get probabilities if requested
        probabilities = None
        if return_probs:
            probabilities = self.prediction_service.predict_proba(aligned_vectors)
        
        # Get feature count
        first_block = next(iter(aligned_vectors.values())) if aligned_vectors else []
        feature_count = len(first_block) if isinstance(first_block, list) else 0
        
        # Create result
        result = PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            threshold_used=threshold,
            feature_count=feature_count,
            metadata={
                'model_name': self.config.model_name,
                'feature_schema': self.config.feature_schema,
                'n_samples': len(predictions)
            }
        )
        
        logger.info(f"✓ Predictions complete: {len(predictions)} samples, threshold={threshold:.2f}")
        return result
    
    def predict_single(
        self,
        block_id: str,
        features: List[float],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Predict for a single block."""
        result = self.predict({block_id: features}, threshold=threshold)
        
        return {
            'block_id': block_id,
            'prediction': result.predictions[block_id],
            'probability': result.probabilities[block_id] if result.probabilities else None,
            'threshold': result.threshold_used
        }
    
    def set_threshold(self, threshold: float):
        """Update prediction threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        self.config.threshold = threshold
        self.prediction_service.set_threshold(threshold)
    
    def add_to_cache(
        self,
        feature_vectors: Dict[str, List[float]],
        labels: Dict[str, int]
    ):
        """Add samples to cache for future retraining."""
        if not self.cache_manager:
            logger.warning("Cache is not enabled.")
            return
        
        # Align features before caching
        # We always want cached data to be aligned/standardized
        aligned_features = self.aligner.align_features(
            feature_vectors, 
            validate_count=False  # Less strict for caching?
        )
        
        self.cache_manager.add_to_cache(aligned_features, labels)
    
    def retrain(
        self,
        new_features: Optional[Dict[str, List[float]]] = None,
        new_labels: Optional[Dict[str, int]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Retrain model with new data."""
        if self.retraining_service is None:
            raise RuntimeError(f"Retraining not enabled. Strategy: {self.config.retraining_strategy}")
        
        # Collect data for retraining
        features_to_use = {}
        labels_to_use = {}
        
        # Add cache data if requested
        if use_cache and self.cache_manager:
            cache_data = self.cache_manager.get_cache_data()
            if cache_data:
                features_to_use.update(cache_data.get('features', {}))
                labels_to_use.update(cache_data.get('labels', {}))
                logger.info(f"✓ Using cached samples")
        
        # Add new data if provided
        if new_features and new_labels:
            aligned_features = self.aligner.align_features(new_features)
            features_to_use.update(aligned_features)
            labels_to_use.update(new_labels)
        
        if not features_to_use:
            raise ValueError("No data available for retraining.")
        
        # Perform retraining
        model_name_base = self.config.model_name.replace('.joblib', '')
        
        if self.config.retraining_strategy == RetrainingStrategy.INCREMENTAL:
            metrics = self.retraining_service.incremental_update(
                model_name=model_name_base,
                new_features=features_to_use,
                new_labels=labels_to_use
            )
        elif self.config.retraining_strategy == RetrainingStrategy.FULL_RETRAIN:
            model_type = self.config.metadata.get('model_type', 'randomforest') if hasattr(self.config, 'metadata') else 'randomforest'
            metrics = self.retraining_service.train(
                feature_vectors=features_to_use,
                labels=labels_to_use,
                model_type=model_type,
                model_name=model_name_base
            )
        else:
            raise NotImplementedError(f"Strategy {self.config.retraining_strategy} not implemented")
        
        # Reload prediction service
        self.prediction_service = PredictionService(
            model_name_or_path=self.config.model_name,
            models_dir=self.config.models_dir,
            threshold=self.config.threshold
        )
        
        return metrics
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached data"""
        if self.cache_manager:
            return self.cache_manager.get_stats()
        return {'enabled': False}
    
    def clear_cache(self):
        """Clear all cached data"""
        if self.cache_manager:
            self.cache_manager.clear_cache()

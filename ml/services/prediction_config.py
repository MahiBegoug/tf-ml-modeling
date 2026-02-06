"""
Prediction Configuration
=======================

Configuration data structures for the Unified Prediction Layer.
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

class RetrainingStrategy(Enum):
    """Supported retraining strategies"""
    NONE = "none"  # No retraining
    INCREMENTAL = "incremental"  # Incremental learning (partial_fit, warm_start, etc.)
    FULL_RETRAIN = "full_retrain"  # Complete retraining from scratch
    BATCH_UPDATE = "batch_update"  # Batch updates at intervals
    CUSTOM = "custom"  # User-defined strategy


@dataclass
class PredictionConfig:
    """Configuration for prediction layer"""
    
    # Model configuration
    model_name: str
    models_dir: str = "models"
    
    # Feature configuration
    feature_schema: Optional[str] = None  # e.g., "lightgbm_features.csv"
    feature_order: Optional[List[str]] = None  # Explicit feature order
    auto_align_features: bool = True  # Automatically align features to model order
    
    # Prediction configuration
    threshold: float = 0.5  # Default prediction threshold
    return_probabilities: bool = True  # Return probabilities along with predictions
    
    # Retraining configuration
    retraining_strategy: RetrainingStrategy = RetrainingStrategy.NONE
    enable_cache: bool = False  # Use cache for historical data
    cache_dir: Optional[str] = None  # Directory for cache storage
    
    # Advanced options
    validate_features: bool = True  # Validate feature count and order
    strict_mode: bool = False  # Fail on any validation error
    
    # Metadata used for retraining
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration"""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {self.threshold}")
        
        if self.enable_cache and not self.cache_dir:
            self.cache_dir = os.path.join(self.models_dir, "cache")

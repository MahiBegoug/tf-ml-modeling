"""
Prediction Result
================

Data structure for prediction results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class PredictionResult:
    """Result from prediction"""
    
    predictions: Dict[str, int]  # block_id -> prediction (0 or 1)
    probabilities: Optional[Dict[str, float]] = None  # block_id -> probability
    threshold_used: float = 0.5
    feature_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

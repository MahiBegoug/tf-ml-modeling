"""
Feature Aligner
==============

Handles the alignment of feature vectors to match a specific schema or model order.
"""

from typing import Dict, List, Optional
from ml.utils.feature_schema import load_feature_schema
from ml.utils.logger import logger

class FeatureAligner:
    """
    Manages feature schema loading and aligns input vectors to that schema.
    """
    
    def __init__(self, 
                 feature_schema_path: Optional[str] = None,
                 feature_order: Optional[List[str]] = None,
                 model_feature_order: Optional[List[str]] = None):
        """
        Initialize feature aligner.
        Priority: schema_path > feature_order > model_feature_order
        """
        self.feature_names = None
        
        if feature_schema_path:
            self.feature_names = load_feature_schema(feature_schema_path)
            logger.info(f"✓ FeatureAligner: Loaded schema {feature_schema_path} ({len(self.feature_names)} features)")
        elif feature_order:
            self.feature_names = feature_order
            logger.info(f"✓ FeatureAligner: Used explicit feature order ({len(self.feature_names)} features)")
        elif model_feature_order:
            self.feature_names = model_feature_order
            logger.info(f"✓ FeatureAligner: Inherited model feature order ({len(self.feature_names)} features)")
            
    def align_features(self, 
                       feature_vectors: Dict[str, List[float]],
                       validate_count: bool = True,
                       strict_mode: bool = False) -> Dict[str, List[float]]:
        """
        Align feature vectors to match the expected order.
        """
        if self.feature_names is None:
            # Nothing to align against
            return feature_vectors
            
        # Check if features are already in list format (optimized path)
        first_val = next(iter(feature_vectors.values())) if feature_vectors else None
        if isinstance(first_val, list):
            if validate_count:
                self._validate_count(len(first_val), strict_mode)
            return feature_vectors
            
        # Alignment logic for dictionary inputs
        aligned_vectors = {}
        for block_id, features_dict in feature_vectors.items():
            if isinstance(features_dict, dict):
                # Reorder according to feature_names
                # Missing features default to 0.0
                aligned_vector = [features_dict.get(feat, 0.0) for feat in self.feature_names]
                aligned_vectors[block_id] = aligned_vector
            else:
                # Should fallback or error? Assume list if not dict
                aligned_vectors[block_id] = features_dict
                
        logger.info(f"✓ Aligned {len(aligned_vectors)} vectors to schema")
        return aligned_vectors

    def _validate_count(self, actual_count: int, strict_mode: bool):
        expected_count = len(self.feature_names)
        if actual_count != expected_count:
            msg = (f"Feature count mismatch! Expected {expected_count}, "
                   f"got {actual_count}.")
            if strict_mode:
                raise ValueError(msg)
            else:
                logger.warning(msg)

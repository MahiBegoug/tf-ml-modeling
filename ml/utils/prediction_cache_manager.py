"""
Prediction Cache Manager
=======================

Handles caching of feature vectors and labels for future retraining.
"""

import os
import joblib
from datetime import datetime
from typing import Dict, List, Any, Optional
from ml.utils.logger import logger

class PredictionCacheManager:
    """
    Manages loading, saving, and updating the prediction cache.
    """
    
    def __init__(self, model_name: str, cache_dir: str):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache = None
        self._initialize_cache()
        
    def _initialize_cache(self):
        """Initialize or load cache from disk."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        cache_file = self._get_cache_file_path()
        
        if os.path.exists(cache_file):
            try:
                self.cache = joblib.load(cache_file)
                logger.info(f"✓ Loaded cache: {len(self.cache.get('features', {}))} samples")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}. Starting fresh.")
                self.clear_cache()
        else:
            self.clear_cache(save=False)
            logger.info("✓ Initialized new cache")

    def _get_cache_file_path(self) -> str:
        return os.path.join(self.cache_dir, f"{self.model_name}_cache.joblib")

    def add_to_cache(self, 
                     aligned_features: Dict[str, List[float]], 
                     labels: Dict[str, int]):
        """
        Add samples to cache.
        
        Args:
            aligned_features: Feature vectors (already aligned to schema)
            labels: Labels for the features
        """
        if self.cache is None:
            self._initialize_cache()
            
        self.cache['features'].update(aligned_features)
        self.cache['labels'].update(labels)
        self.cache['metadata']['last_updated'] = datetime.now().isoformat()
        
        self.save_cache()
        logger.info(f"✓ Added {len(aligned_features)} samples to cache")

    def save_cache(self):
        """Save cache to disk."""
        if self.cache is not None:
            try:
                joblib.dump(self.cache, self._get_cache_file_path())
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")

    def get_cache_data(self) -> Dict[str, Any]:
        """Return the raw cache data."""
        return self.cache if self.cache else {}

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about cached data."""
        if self.cache is None:
            return {'enabled': True, 'item_count': 0}
            
        features = self.cache.get('features', {})
        labels = self.cache.get('labels', {})
        
        return {
            'enabled': True,
            'total_samples': len(features),
            'defective': sum(labels.values()),
            'clean': len(labels) - sum(labels.values()),
            'last_updated': self.cache['metadata'].get('last_updated', 'Never'),
            'created': self.cache['metadata'].get('created', 'Unknown')
        }

    def clear_cache(self, save: bool = True):
        """Clear all cached data."""
        self.cache = {
            'features': {},
            'labels': {},
            'metadata': {
                'created': datetime.now().isoformat(),
                'model_name': self.model_name
            }
        }
        if save:
            self.save_cache()
            logger.info("✓ Cache cleared")

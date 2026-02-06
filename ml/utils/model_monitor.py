"""
Model Monitor
============

Utilities for monitoring model performance history.
"""

import os
import json
from typing import List, Dict, Optional

class ModelMonitor:
    """
    Monitors model training and update history.
    """
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir

    def get_model_history(self, model_name: str) -> List[Dict]:
        """
        Get training/update history for a model.
        
        Args:
            model_name: Name of the model (without directory)
            
        Returns:
            List of historical metrics
        """
        # Strip .joblib extension if present to find history file
        base_name = model_name.replace('.joblib', '')
        metadata_path = os.path.join(self.models_dir, f"{base_name}_history.json")
        
        if not os.path.exists(metadata_path):
            return []
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading history file {metadata_path}: {e}")
            return []
    
    def print_model_history(self, model_name: str):
        """
        Print model training/update history in a readable format.
        
        Args:
            model_name: Name of the model
        """
        history = self.get_model_history(model_name)
        
        if not history:
            print(f"No history found for model: {model_name}")
            return
        
        print("=" * 70)
        print(f"MODEL HISTORY: {model_name}")
        print("=" * 70)
        print(f"{'#':<4} {'Type':<20} {'Samples':<10} {'MCC':<8} {'G-Mean':<8} {'Timestamp':<20}")
        print("-" * 70)
        
        for i, entry in enumerate(history, 1):
            n_samples = entry.get('n_samples', entry.get('n_new_samples', 0))
            print(
                f"{i:<4} "
                f"{entry.get('training_type', 'unknown'):<20} "
                f"{n_samples:<10} "
                f"{entry.get('mcc', 0):.3f}    "
                f"{entry.get('gmean', 0):.3f}    "
                f"{entry.get('timestamp', 'unknown')[:19]}"
            )

    def log_training(self, model_name: str, metrics: Dict, append: bool = True):
        """
        Log training metrics to history file.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics to save
            append: Whether to append to existing history (default True)
        """
        base_name = model_name.replace('.joblib', '')
        metadata_path = os.path.join(self.models_dir, f"{base_name}_history.json")
        
        history = []
        if append and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    history = json.load(f)
            except Exception:
                # If error reading (e.g. corrupt), start fresh
                history = []
        
        history.append(metrics)
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Error saving history to {metadata_path}: {e}")

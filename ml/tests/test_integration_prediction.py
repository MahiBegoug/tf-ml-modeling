import os
import sys
import unittest
import shutil
import tempfile
import numpy as np

# Ensure we can import infrastructure
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ml.services.unified_prediction_layer import UnifiedPredictionLayer
from ml.services.prediction_config import PredictionConfig
from ml.services.retraining_service import RetrainingService

class TestStandaloneML(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.models_dir)
        
        # Create a dummy model for testing
        self.retrainer = RetrainingService(self.models_dir)
        
        # Create synthetic data
        self.features = {
            f"block_{i}": np.random.rand(10).tolist() for i in range(20)
        }
        self.labels = {
            f"block_{i}": i % 2 for i in range(20)
        }
        
        # Train initial model
        self.retrainer.train(
            feature_vectors=self.features,
            labels=self.labels,
            model_type="randomforest",
            model_name="test_model"
        )
        
        # Configure Prediction Layer
        self.config = PredictionConfig(
            model_name="test_model",
            models_dir=self.models_dir,
            auto_align_features=True
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_end_to_end_prediction(self):
        """Test that the standalone ML component can predict without external deps."""
        layer = UnifiedPredictionLayer(self.config)
        
        # Predict
        result = layer.predict(self.features)
        
        # Verify
        from ml.services.prediction_result import PredictionResult
        self.assertTrue(isinstance(result, PredictionResult))
        self.assertTrue(isinstance(result.predictions, dict))
        self.assertEqual(len(result.predictions), len(self.features))
        
        print(f"\n✓ Prediction successful: {len(result.predictions)} items")

    def test_model_monitoring(self):
        """Test that history is recorded."""
        # Train another model to generate history
        self.retrainer.train(
            self.features, self.labels, model_name="monitor_test"
        )
        
        from ml.utils.model_monitor import ModelMonitor
        monitor = ModelMonitor(self.models_dir)
        history = monitor.get_model_history("monitor_test")
        
        self.assertTrue(len(history) > 0)
        self.assertEqual(history[0]['model_type'], 'randomforest')
        print(f"\n✓ History verification successful: {len(history)} entries")

if __name__ == "__main__":
    unittest.main()

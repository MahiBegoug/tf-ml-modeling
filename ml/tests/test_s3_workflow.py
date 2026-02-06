import unittest
import os
import sys
import shutil
import tempfile
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from download_models import download_s3_folder
from ml.services.prediction_config import PredictionConfig
from ml.services.unified_prediction_layer import UnifiedPredictionLayer

class TestS3Workflow(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for "downloaded" models
        self.test_dir = tempfile.mkdtemp()
        self.bucket_name = "test-bucket"
        self.prefix = "models/"
        
    def tearDown(self):
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    @patch('boto3.client')
    def test_download_s3_folder(self, mock_boto_client):
        """Test that download_models.py correctly 'downloads' files from S3."""
        
        # Mock S3 Client and Paginator
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.get_paginator.return_value = mock_paginator
        
        # Simulate S3 listing
        # Structure: models/trained_models/model.joblib
        mock_paginator.paginate.return_value = [
            {
                'Contents': [
                    {'Key': 'models/trained_models/dummy_model.joblib'},
                    {'Key': 'models/model_features/schema.json'},
                    {'Key': 'models/'} # Folder holder
                ]
            }
        ]

        # run the download function
        success = download_s3_folder(
            bucket_name=self.bucket_name,
            s3_prefix=self.prefix,
            local_dir=self.test_dir
        )
        
        self.assertTrue(success)
        
        # Verify download_file was called for the files (not the folder)
        expected_calls = [
            unittest.mock.call(
                self.bucket_name, 
                'models/trained_models/dummy_model.joblib', 
                str(Path(self.test_dir) / 'trained_models/dummy_model.joblib')
            ),
            unittest.mock.call(
                self.bucket_name, 
                'models/model_features/schema.json', 
                str(Path(self.test_dir) / 'model_features/schema.json')
            )
        ]
        mock_s3.download_file.assert_has_calls(expected_calls, any_order=True)
        
        # Verify call count (should be 2 files)
        self.assertEqual(mock_s3.download_file.call_count, 2)

    def test_dummy_end_to_end_flow(self):
        """
        Simulate the entire flow:
        1. Models are 'downloaded' (we create dummy files manually here to simulate it)
        2. PredictionLayer is initialized pointing to these models
        3. Prediction is run
        """
        
        # 1. Simulate download by creating dummy files
        models_dir = Path(self.test_dir) / "trained_models"
        features_dir = Path(self.test_dir) / "model_features"
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)
        
        # Create a dummy model file (UnifiedPredictionLayer needs to find it)
        dummy_model_path = models_dir / "defect_predictor_v1.joblib"
        with open(dummy_model_path, 'w') as f:
            f.write("dummy model content")
            
        print(f"Created simulated model at {dummy_model_path}")

        # 2. Configure Prediction
        # We assume the config will point to our temp dir
        config = PredictionConfig(
            model_name=str(dummy_model_path), # Point directly to our dummy
            models_dir=str(models_dir),
            threshold=0.5
        )
        
        # 3. Initialize Layer
        # We need to mock the ModelLoader because we wrote garbage to the joblib file
        # Also need to mock FeatureAligner because it might try to load schema
        with patch('ml.utils.model_loader.ModelLoader.load_model') as mock_load, \
             patch('ml.utils.feature_aligner.FeatureAligner.align_features') as mock_align, \
             patch('ml.utils.feature_aligner.load_feature_schema') as mock_load_schema:
            
            # Setup Mock Model
            mock_model_obj = MagicMock()
            mock_model_obj.n_features_in_ = 3  # Important for FeatureValidator
            mock_load.return_value = mock_model_obj
            
            # Mock prediction logic (return numpy arrays)
            # PredictionService logic: (probs[:, 1] >= threshold)
            mock_model_obj.predict.return_value = np.array([0]) 
            mock_model_obj.predict_proba.return_value = np.array([[0.8, 0.2]]) 
            
            # Mock FeatureAligner to return a aligned DICTIONARY
            # UnifiedPredictionLayer expects dict to pass to service.predict(dict)
            mock_align.return_value = {"block_1": [1, 2, 3]} 
            
            layer = UnifiedPredictionLayer(config)
            
            # 4. Predict
            features = {"block_1": [1, 2, 3]} # Dummy features
            result = layer.predict(features)
            
            self.assertIsNotNone(result)
            # The result.predictions should be a dict
            self.assertEqual(result.predictions, {'block_1': 0})
            print("Prediction successfully run with simulated S3-downloaded model path")

if __name__ == '__main__':
    unittest.main()

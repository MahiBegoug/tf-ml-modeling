
import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from pathlib import Path
import datetime

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ml.utils.model_saver import ModelSaver

class TestS3Upload(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('ml.utils.s3_utils.upload_file_to_s3')
    @patch.dict(os.environ, {
        'S3_BUCKET': 'test-bucket', 
        'S3_MODEL_PREFIX': 'models/',
        'AWS_REGION': 'us-east-1'
    })
    def test_save_model_uploads_with_version(self, mock_upload):
        """Test that saving a model triggers an S3 upload with a version timestamp."""
        
        # Setup
        saver = ModelSaver(self.models_dir)
        # MagicMock cannot be pickled by joblib, use a simple dict as dummy model
        mock_model = {"model_type": "dummy", "params": {}}
        model_name = "defect_predictor"
        
        # Mock upload to return True
        mock_upload.return_value = True
        
        # Action
        saved_path = saver.save_model(mock_model, model_name)
        
        # Verify
        self.assertTrue(os.path.exists(saved_path))
        
        # Check if upload was called
        self.assertTrue(mock_upload.called)
        
        # Inspect arguments
        args, kwargs = mock_upload.call_args
        
        # Expected args: local_path, bucket_name, s3_prefix, object_name, aws_region
        # Note: upload_file_to_s3 signature: (local_path, bucket_name, s3_prefix, object_name=None, aws_region=None)
        
        # Verify kwargs
        self.assertEqual(kwargs['bucket_name'], 'test-bucket')
        self.assertEqual(kwargs['s3_prefix'], 'models/trained_models')
        self.assertEqual(kwargs['aws_region'], 'us-east-1')
        
        # Verify versioned name
        object_name = kwargs['object_name']
        self.assertTrue(object_name.startswith("defect_predictor_v"))
        self.assertTrue(object_name.endswith(".joblib"))
        
        # Basic check for timestamp format (length check logic)
        # "defect_predictor_vYYYYMMDDHHMMSS.joblib"
        # defect_predictor (16) + _v (2) + timestamp (14) + .joblib (7) = 39 chars approx
        self.assertRegex(object_name, r"defect_predictor_v\d{14}\.joblib")
        
        print(f"\n✓ Verified upload call for: {object_name}")

if __name__ == '__main__':
    unittest.main()

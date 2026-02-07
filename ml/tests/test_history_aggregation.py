
import unittest
import os
import shutil
import pandas as pd
from unittest.mock import patch, MagicMock
from ml.scripts.aggregate_history import aggregate_prediction_history

class TestHistoryAggregation(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = "test_aggregation_temp"
        self.output_file = "test_history_master.csv"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create dummy CSVs
        df1 = pd.DataFrame({'feature1': [1, 2], 'prediction': [0, 1]})
        df2 = pd.DataFrame({'feature1': [3, 4], 'prediction': [1, 0]})
        
        df1.to_csv(os.path.join(self.test_dir, "predictions_abc123_20231027000000.csv"), index=False)
        df2.to_csv(os.path.join(self.test_dir, "predictions_def456_20231028000000.csv"), index=False)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    @patch('ml.scripts.aggregate_history.download_s3_folder')
    @patch('ml.scripts.aggregate_history.upload_file_to_s3')
    @patch('shutil.rmtree')
    def test_aggregation_logic(self, mock_rmtree, mock_upload, mock_download):
        # Mock download to do nothing (files already created in setUp)
        # We need to trick the script to use our setUp directory instead of downloading
        # But script takes temp_dir as arg.
        
        mock_download.return_value = True
        mock_upload.return_value = True
        
        result_path = aggregate_prediction_history(
            bucket_name="test-bucket",
            temp_dir=self.test_dir,
            output_file=self.output_file
        )
        
        self.assertIsNotNone(result_path)
        self.assertTrue(os.path.exists(result_path))
        
        # Verify Content
        df_master = pd.read_csv(result_path)
        
        # Should have 4 rows (2 from each file)
        self.assertEqual(len(df_master), 4)
        
        # Should have new columns
        self.assertIn("commit_hash", df_master.columns)
        self.assertIn("timestamp", df_master.columns)
        
        # Verify Metadata parsing
        # Check first file's rows
        row0 = df_master.iloc[0]
        # Depending on order, check values.
        # Commit hash should be abc123 or def456
        self.assertTrue(row0['commit_hash'] in ['abc123', 'def456'])

if __name__ == '__main__':
    unittest.main()

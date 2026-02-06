# Defect Prediction ML Package

This is a standalone Machine Learning component for Terraform Defect Prediction.

## Features
- **Unified Prediction Layer**: One entry point for all predictions.
- **Model Agnostic**: Supports RandomForest, LightGBM, SGD, etc.
- **Retraining**: Incremental or full retraining strategies.
- **Feature Alignment**: Automatically handles feature schema differences.
- **Monitoring**: Logs model performance history.

## Installation

You can install this package via pip:

```bash
pip install .
```

Or install dependencies manually:

```bash
pip install -r requirements.txt
```

## Downloading Pre-trained Models from S3

This package supports automatic downloading of pre-trained models from AWS S3.

### Setup

1. **Configure AWS Credentials**

   Create a `.env` file in the project root (use `.env.example` as template):

   ```bash
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-east-1
   S3_BUCKET=your-bucket-name
   S3_MODEL_PREFIX=models/
   ```

2. **Download Models**

   **Option A: Using Docker Compose** (Automatic on startup)
   
   ```bash
   docker-compose up
   ```
   
   Models will be automatically downloaded if `S3_BUCKET` is set.

   **Option B: Manual Download**
   
   ```bash
   python download_models.py --bucket your-bucket --prefix models/ --output ./pre_trained_defect_models
   ```

   **Option C: Using Environment Variables**
   
   ```bash
   export S3_BUCKET=your-bucket-name
   export S3_MODEL_PREFIX=models/
   python download_models.py
   ```

### CI/CD Configuration (GitHub Actions)

To enable S3 downloads in your GitHub Actions workflow, add the following secrets to your repository settings (**Settings** > **Secrets and variables** > **Actions**):

| Secret Name | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | Your AWS Access Key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS Secret Key |
| `AWS_REGION` | AWS Region (e.g., `us-east-1`) |
| `S3_BUCKET` | Your S3 Bucket Name |

### S3 Bucket Structure

Your S3 bucket should be organized as follows:

```
s3://your-bucket-name/
└── models/
    ├── trained_models/
    │   ├── random_forest_model.joblib
    │   ├── lightgbm_model.joblib
    │   └── ...
    └── model_features/
        ├── feature_schema.json
        └── ...
```

## Usage

### Basic Prediction

```python
from infrastructure.ml.services.unified_prediction_layer import UnifiedPredictionLayer
from infrastructure.ml.services.prediction_config import PredictionConfig

# 1. Configure
config = PredictionConfig(
    model_name="my_model",
    models_dir="path/to/models",
    threshold=0.5
)

# 2. Initialize
layer = UnifiedPredictionLayer(config)

# 3. Predict
# features = {"block_1": [0.5, 0.1, ...], ...}
result = layer.predict(features)

print(result.predictions)  # {'block_1': 0, ...}
```

### Retraining

```python
from infrastructure.ml.services.prediction_config import RetrainingStrategy

config = PredictionConfig(
    model_name="my_model",
    retraining_strategy=RetrainingStrategy.INCREMENTAL
    # ...
)

layer = UnifiedPredictionLayer(config)

# Retrain with new data
layer.retrain(new_features, new_labels)
```

## Structure

- `services/`: Core logic (Prediction, Retraining, Config).
- `utils/`: Helpers (Feature Aligner, Model Loader, Monitor).
- `models/`: Model definitions and factories.
- `pretrained_models/`: Included pre-trained models (e.g., `defect_predictor_v1.joblib`).
- `features/`: Feature schemas.

## Pretrained Models

The package includes sample pretrained models in `pretrained_models/`. You can use them directly:

```python
import os
from infrastructure.ml.services.prediction_config import PredictionConfig

# Get path to pretrained model
pkg_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(pkg_dir, "pretrained_models", "defect_predictor_v1.joblib")

config = PredictionConfig(
    model_name=model_path,  # can pass full path
    # ...
)
```
## Testing

Run the integration test:

```bash
python -m unittest tests/test_integration_prediction.py
```
(Make sure to set PYTHONPATH if running from inside the directory)

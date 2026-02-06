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

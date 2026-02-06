"""
User Training Service
====================

Allows users to train models from scratch using their own CSV datasets.
Handles data loading, automatic feature selection, model training, and artifact saving.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

from ml.services.retraining_service_with_scaler import RetrainingServiceWithScaler
from ml.utils.logger import logger

class UserTrainingService:
    """
    Orchestrates the training process from a raw CSV file.
    
    Workflow:
    1. Load CSV
    2. Preprocess (Encode targets, handle non-numeric)
    3. Feature Selection (Top K features)
    4. Train Model (using RetrainingServiceWithScaler)
    5. Save Artifacts (Model + Feature Schema)
    """
    
    def __init__(self, models_dir: str = "models", features_dir: str = "features"):
        self.models_dir = models_dir
        self.features_dir = features_dir
        
        # Ensure directories exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)
        
        # Initialize internal retraining service
        self.retrainer = RetrainingServiceWithScaler(models_dir, use_scaler=True)

    def train_from_csv(self,
                       csv_path: str,
                       target_col: str = "defect",
                       model_type: str = "randomforest",
                       top_k_features: int = 20,
                       tune_hyperparameters: bool = False,
                       tune_n_jobs: int = 1, # Default to 1 for safety on Windows
                       output_name: Optional[str] = None,
                       custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train a model from scratch using a CSV file.
        
        Args:
            csv_path: Path to training CSV
            target_col: Name of the target column
            model_type: One of ['randomforest', 'lightgbm', 'logisticreg', 'naivebayes', 'svm', 'mlp']
            top_k_features: Number of features to select
            output_name: Name for saved artifacts (default: model_type)
            
        Returns:
            Dict containing training metrics and artifact paths.
        """
        if not output_name:
            output_name = model_type

        logger.info(f"[Start] Starting user training for {model_type}...")
        logger.info(f"[Load] Loading data from {csv_path}")

        # 1. Load Data
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV.")

        # 2. Preprocessing
        # Separate X and y
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Handle non-numeric features (simple drop for now, or basic encoding)
        # For robustness, let's keep only numeric
        X_numeric = X.select_dtypes(include=[np.number])
        dropped_cols = list(set(X.columns) - set(X_numeric.columns))
        if dropped_cols:
            logger.warning(f"⚠️ Dropped non-numeric columns: {dropped_cols}")
        X = X_numeric

        # Encode target if it's text (e.g., 'Clean', 'Buggy')
        if y.dtype == 'object' or y.dtype.name == 'category':
            logger.info("Encoding target labels...")
            le = LabelEncoder()
            y = le.fit_transform(y)
            # Assumption: 0=Negative, 1=Positive. 
            # If user has ['Buggy', 'Clean'], check mapping.
            # Ideally, user should provide 0/1. Warning logged.
            logger.warning(f"Target encoded. Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # 3. Feature Selection
        logger.info(f"🔍 Performing feature selection (Top {top_k_features})...")
        
        # A. Variance Threshold (Remove constants)
        selector_var = VarianceThreshold(threshold=0)
        X_var = selector_var.fit_transform(X)
        selected_indices = selector_var.get_support(indices=True)
        X_filtered = X.iloc[:, selected_indices]
        
        logger.info(f"✓ Variance logic removed {X.shape[1] - X_filtered.shape[1]} constant features")

        # B. Select K Best (ANOVA F-value)
        k = min(top_k_features, X_filtered.shape[1])
        selector_k = SelectKBest(f_classif, k=k)
        X_new = selector_k.fit_transform(X_filtered, y)
        
        # Get selected feature names
        mask = selector_k.get_support()
        selected_features = X_filtered.columns[mask].tolist()
        
        logger.info(f"✓ Selected {len(selected_features)} features: {selected_features}")

        # 4. Save Feature Schema
        schema_path = os.path.join(self.features_dir, f"{output_name}_features.csv")
        with open(schema_path, "w") as f:
            f.write("\n".join(selected_features))
        logger.info(f"💾 Saved feature schema to {schema_path}")

        # 5. Model Tuning (Optional)
        if tune_hyperparameters:
            logger.info("🔧 Starting Hyperparameter Tuning...")
            try:
                from ml.services.hyperparameter_tuning_service import HyperparameterTuningService
                tuner = HyperparameterTuningService(n_jobs=tune_n_jobs, cv=5)
                
                # Tuner expects arrays
                tune_results = tuner.tune_and_train(X_new, y, model_type)
                
                # Extract best params (remove 'classifier__' prefix if present)
                best_params = tune_results.get("best_params", {})
                custom_params = {}
                for k, v in best_params.items():
                    clean_key = k.replace("classifier__", "")
                    custom_params[clean_key] = v
                    
                logger.info(f"✓ Tuning complete. Using params: {custom_params}")
            except Exception as e:
                logger.error(f"Hyperparameter tuning failed: {e}. Proceeding with defaults.")
                import traceback
                traceback.print_exc()

        # 6. Train Model
        # Convert X_new back to Dict[str, List[float]] for RetrainingService
        feature_vectors = {
            f"row_{i}": row.tolist() 
            for i, row in enumerate(X_new)
        }
        labels = {
            f"row_{i}": int(label)
            for i, label in enumerate(y)
        }

        # Initialize MLflow
        import mlflow
        mlflow.set_experiment("UserTraining")

        with mlflow.start_run() as run:
            logger.info(f"🚀 Started MLflow run: {run.info.run_id}")
            
            # Log Parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_samples", len(X_new))
            mlflow.log_param("n_features", len(selected_features))
            if custom_params:
                mlflow.log_params(custom_params)

            metrics = self.retrainer.train(
                feature_vectors=feature_vectors,
                labels=labels,
                model_type=model_type,
                model_name=output_name,
                custom_params=custom_params
            )

            metrics["feature_schema_path"] = schema_path
            metrics["selected_features"] = selected_features
            
            # Log Metrics
            # Flatten metrics if nested (though train returns flat usually)
            safe_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(safe_metrics)
            
            # Log Model Artifact
            # We log the saved joblib file to preserve the Bundle structure (model + scaler)
            model_path = metrics.get('model_path')
            if model_path and os.path.exists(model_path):
                mlflow.log_artifact(model_path)
                mlflow.log_artifact(schema_path)
            
            logger.info(f"✅ MLflow logging complete.")
        
        return metrics

    def predict_from_csv(self,
                        input_csv: str,
                        model_name: str,
                        output_csv: Optional[str] = None,
                        threshold: float = 0.5) -> pd.DataFrame:
        """
        Run predictions on a CSV file using a trained model.
        
        Args:
            input_csv: Path to input CSV
            model_name: Name of the model to usage (must be in models_dir)
            output_csv: Optional path to save predictions
            threshold: Probability threshold for positive class (default: 0.5)
            
        Returns:
            DataFrame with original data + predictions
        """
        logger.info(f"🔮 Predicting using model '{model_name}' on '{input_csv}' [Threshold: {threshold}]")
        
        # 1. Load Model Bundle
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        try:
            import joblib
            bundle = joblib.load(model_path)
            model = bundle['model'] if isinstance(bundle, dict) else bundle
            scaler = bundle.get('scaler') if isinstance(bundle, dict) else None
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

        # 2. Load Feature Schema
        schema_path = os.path.join(self.features_dir, f"{model_name}_features.csv")
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                selected_features = [line.strip() for line in f.readlines()]
            logger.info(f"✓ Loaded feature schema: {len(selected_features)} features")
        else:
            logger.warning(f"Feature schema not found at {schema_path}. Assuming all columns in CSV are features.")
            selected_features = None

        # 3. Load Data
        df = pd.read_csv(input_csv)
        
        # 4. Align Data
        if selected_features:
            # Check missing
            missing = [f for f in selected_features if f not in df.columns]
            if missing:
                raise ValueError(f"Input CSV matches schema but missing columns: {missing}")
            X = df[selected_features].values
        else:
            # Use all numeric
            X = df.select_dtypes(include=[np.number]).values
        
        # 5. Scale
        if scaler:
            X = scaler.transform(X)
            
        # 6. Predict with Threshold
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= threshold).astype(int)
        else:
            # Fallback for models without proba (unlikely in this stack)
            preds = model.predict(X)
            probs = [0.0] * len(X)
        
        # 7. Create Result
        results = df.copy()
        results["prediction"] = preds
        results["probability"] = probs
        
        if output_csv:
            results.to_csv(output_csv, index=False)
            logger.info(f"💾 Predictions saved to {output_csv}")
            
        return results
        """
        Train a model from scratch using a CSV file.
        
        Args:
            csv_path: Path to training CSV
            target_col: Name of the target column
            model_type: One of ['randomforest', 'lightgbm', 'logisticreg', 'naivebayes', 'svm', 'mlp']
            top_k_features: Number of features to select
            output_name: Name for saved artifacts (default: model_type)
            
        Returns:
            Dict containing training metrics and artifact paths.
        """
        if not output_name:
            output_name = model_type

        logger.info(f"[Start] Starting user training for {model_type}...")
        logger.info(f"[Load] Loading data from {csv_path}")

        # 1. Load Data
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV.")

        # 2. Preprocessing
        # Separate X and y
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Handle non-numeric features (simple drop for now, or basic encoding)
        # For robustness, let's keep only numeric
        X_numeric = X.select_dtypes(include=[np.number])
        dropped_cols = list(set(X.columns) - set(X_numeric.columns))
        if dropped_cols:
            logger.warning(f"⚠️ Dropped non-numeric columns: {dropped_cols}")
        X = X_numeric

        # Encode target if it's text (e.g., 'Clean', 'Buggy')
        if y.dtype == 'object' or y.dtype.name == 'category':
            logger.info("Encoding target labels...")
            le = LabelEncoder()
            y = le.fit_transform(y)
            # Assumption: 0=Negative, 1=Positive. 
            # If user has ['Buggy', 'Clean'], check mapping.
            # Ideally, user should provide 0/1. Warning logged.
            logger.warning(f"Target encoded. Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # 3. Feature Selection
        logger.info(f"🔍 Performing feature selection (Top {top_k_features})...")
        
        # A. Variance Threshold (Remove constants)
        selector_var = VarianceThreshold(threshold=0)
        X_var = selector_var.fit_transform(X)
        selected_indices = selector_var.get_support(indices=True)
        X_filtered = X.iloc[:, selected_indices]
        
        logger.info(f"✓ Variance logic removed {X.shape[1] - X_filtered.shape[1]} constant features")

        # B. Select K Best (ANOVA F-value)
        k = min(top_k_features, X_filtered.shape[1])
        selector_k = SelectKBest(f_classif, k=k)
        X_new = selector_k.fit_transform(X_filtered, y)
        
        # Get selected feature names
        mask = selector_k.get_support()
        selected_features = X_filtered.columns[mask].tolist()
        
        logger.info(f"✓ Selected {len(selected_features)} features: {selected_features}")

        # 4. Save Feature Schema
        schema_path = os.path.join(self.features_dir, f"{output_name}_features.csv")
        with open(schema_path, "w") as f:
            f.write("\n".join(selected_features))
        logger.info(f"💾 Saved feature schema to {schema_path}")

        # 5. Model Tuning (Optional)
        pass # Placeholder needed if loop below is empty, but it won't be

        # 5. Model Tuning (Optional)
        if tune_hyperparameters:
            logger.info("🔧 Starting Hyperparameter Tuning...")
            try:
                from ml.services.hyperparameter_tuning_service import HyperparameterTuningService
                tuner = HyperparameterTuningService(n_jobs=tune_n_jobs, cv=5)
                
                # Tuner expects arrays
                tune_results = tuner.tune_and_train(X_new, y, model_type)
                
                # Extract best params (remove 'classifier__' prefix if present)
                best_params = tune_results.get("best_params", {})
                custom_params = {}
                for k, v in best_params.items():
                    clean_key = k.replace("classifier__", "")
                    custom_params[clean_key] = v
                    
                logger.info(f"✓ Tuning complete. Using params: {custom_params}")
            except Exception as e:
                logger.error(f"Hyperparameter tuning failed: {e}. Proceeding with defaults.")
                import traceback
                traceback.print_exc()

        # 6. Train Model
        # Convert X_new back to Dict[str, List[float]] for RetrainingService
        feature_vectors = {
            f"row_{i}": row.tolist() 
            for i, row in enumerate(X_new)
        }
        labels = {
            f"row_{i}": int(label)
            for i, label in enumerate(y)
        }

        metrics = self.retrainer.train(
            feature_vectors=feature_vectors,
            labels=labels,
            model_type=model_type,
            model_name=output_name,
            custom_params=custom_params
        )

        metrics["feature_schema_path"] = schema_path
        metrics["selected_features"] = selected_features
        
        return metrics

    def predict_from_csv(self,
                        input_csv: str,
                        model_name: str,
                        output_csv: Optional[str] = None,
                        threshold: float = 0.5) -> pd.DataFrame:
        """
        Run predictions on a CSV file using a trained model.
        
        Args:
            input_csv: Path to input CSV
            model_name: Name of the model to usage (must be in models_dir)
            output_csv: Optional path to save predictions
            threshold: Probability threshold for positive class (default: 0.5)
            
        Returns:
            DataFrame with original data + predictions
        """
        logger.info(f"🔮 Predicting using model '{model_name}' on '{input_csv}' [Threshold: {threshold}]")
        
        # 1. Load Model Bundle
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        try:
            import joblib
            bundle = joblib.load(model_path)
            model = bundle['model'] if isinstance(bundle, dict) else bundle
            scaler = bundle.get('scaler') if isinstance(bundle, dict) else None
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

        # 2. Load Feature Schema
        schema_path = os.path.join(self.features_dir, f"{model_name}_features.csv")
        selected_features = None

        def load_schema_if_exists(path):
            if os.path.exists(path):
                with open(path, "r") as f:
                    features = [line.strip() for line in f.readlines()]
                # Heuristic: Check for generic "f0", "f1" headers which indicate bad schema
                if len(features) > 0 and features[0] == "f0" and features[1] == "f1":
                    logger.warning(f"⚠️ Ignoring generic schema at {path} (f0, f1 detected).")
                    return None
                return features
            return None

        # A. Try Exact Match
        selected_features = load_schema_if_exists(schema_path)

        # B. Fallback: Remove "_model" (random_forest_model -> random_forest)
        if not selected_features and "_model" in model_name:
            short_name = model_name.replace("_model", "")
            path = os.path.join(self.features_dir, f"{short_name}_features.csv")
            selected_features = load_schema_if_exists(path)
            
            # C. Fallback: Remove Underscores (random_forest -> randomforest)
            if not selected_features:
                clean_name = short_name.replace("_", "")
                path = os.path.join(self.features_dir, f"{clean_name}_features.csv")
                logger.info(f"ℹ️ Checking underscore-stripped fallback: {path}")
                selected_features = load_schema_if_exists(path)

        # 2.5. Validate and Truncate Schema to Match Model's Expected Features
        if selected_features:
            # Remove header row if present (e.g., "Feature")
            if selected_features and selected_features[0] == "Feature":
                selected_features = selected_features[1:]
                logger.info("ℹ️ Removed 'Feature' header from schema")
            
            # Check model's expected feature count
            model_expected_features = None
            if hasattr(model, 'n_features_in_'):
                model_expected_features = model.n_features_in_
            
            if model_expected_features and len(selected_features) > model_expected_features:
                logger.warning(
                    f"⚠️ Schema has {len(selected_features)} features, but model expects {model_expected_features}. "
                    f"Auto-truncating to first {model_expected_features} features."
                )
                selected_features = selected_features[:model_expected_features]
            
            logger.info(f"✓ Final feature schema: {len(selected_features)} features")
        else:
            logger.warning(f"❌ Feature schema not found or invalid. Assuming all columns in CSV are features.")
            selected_features = None

        # 3. Load Data
        df = pd.read_csv(input_csv)
        
        # 4. Align Data
        if selected_features:
            # Check missing
            missing = [f for f in selected_features if f not in df.columns]
            if missing:
                raise ValueError(f"Input CSV matches schema but missing columns: {missing}")
            X = df[selected_features].values
        else:
            # Use all numeric
            X = df.select_dtypes(include=[np.number]).values
        
        # 5. Scale
        if scaler:
            X = scaler.transform(X)
            
        # 6. Predict with Threshold
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= threshold).astype(int)
        else:
            # Fallback for models without proba (unlikely in this stack)
            preds = model.predict(X)
            probs = [0.0] * len(X)
        
        # 7. Create Result
        results = df.copy()
        results["prediction"] = preds
        results["probability"] = probs
        
        if output_csv:
            results.to_csv(output_csv, index=False)
            logger.info(f"💾 Predictions saved to {output_csv}")
            
        return results

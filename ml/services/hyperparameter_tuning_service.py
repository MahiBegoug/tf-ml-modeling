"""
Hyperparameter Tuning Service
============================

Optimizes model hyperparameters using GridSearchCV or RandomizedSearchCV.
Uses user-defined parameter grids.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from ml.services.retraining_service_with_scaler import RetrainingServiceWithScaler
from ml.utils.logger import logger

# Import models
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

class HyperparameterTuningService:
    """
    Service to tune hyperparameters for the supported models.
    """
    
    # User-defined parameter grids
    # Note: Keys are prefixed with 'classifier__' because we will use a Pipeline
    PARAM_GRIDS = {
        "lightgbm": {
            "classifier__learning_rate": [0.1, 0.01, 0.2, 0.3, 0.4],
            "classifier__reg_alpha": [0.1, 0.2, 0.6, 0.7, 0.8], # 'alpha' usually maps to reg_alpha in LGBM
            "classifier__max_depth": [3, 5, 7, 11]
        },
        "extratrees": {
            "classifier__n_estimators": [50, 100, 200, 400, 600],
            "classifier__max_depth": [3, 4, 5, 6, 10],
            "classifier__criterion": ["gini", "entropy", "log_loss"],
            "classifier__min_samples_split": [2, 10, 50, 100],
            "classifier__min_samples_leaf": [1, 2, 3, 4, 5, 50],
            "classifier__max_features": ['sqrt', 'log2', None]
        },
        "randomforest": {
            "classifier__n_estimators": [50, 100, 200, 400, 600],
            "classifier__max_depth": [3, 4, 5, 6, 10],
            "classifier__criterion": ["gini", "entropy", "log_loss"],
            "classifier__min_samples_split": [2, 10, 50, 100],
            "classifier__min_samples_leaf": [1, 2, 3, 4, 5, 50],
            "classifier__max_features": ['sqrt', 'log2']
        },
        "decisiontree": {
            "classifier__criterion": ['gini', 'entropy', 'log_loss'],
            "classifier__splitter": ['best', 'random'],
            "classifier__max_depth": [3, 5, 7, 11, 17, 21],
            "classifier__max_features": ['sqrt', 'log2']
        },
        "logisticreg": {
            "classifier__penalty": ['l2', None],
            "classifier__fit_intercept": [True, False],
            "classifier__max_iter": [100, 10000],
            "classifier__tol": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
            "classifier__solver": ['lbfgs', 'saga'],
            "classifier__C": [0.005, 0.05, 0.5, 1.0]
        },
        "naivebayes": {
            "classifier__var_smoothing": [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
        },
        "dummy": {
            "classifier__strategy": ["stratified"]
        }
    }

    def __init__(self, n_jobs: int = -1, cv: int = 5):
        self.n_jobs = n_jobs
        self.cv = cv

    def tune_and_train(self, 
                       X: np.ndarray, 
                       y: np.ndarray, 
                       model_type: str,
                       scoring: str = 'f1') -> Dict[str, Any]:
        """
        Tune hyperparameters and return the best parameters and metrics.
        
        Args:
            X: aligned features
            y: aligned labels
            model_type: type of model (e.g. 'randomforest')
            scoring: metric to optimize (default: 'f1')
            
        Returns:
            Dict containing 'best_params', 'best_score', 'best_estimator'
        """
        model_type = model_type.lower()
        if model_type not in self.PARAM_GRIDS and model_type != "svm" and model_type != "mlp":
             logger.warning(f"No tuning grid defined for {model_type}. Using defaults.")
             # Fallback logic could go here, for now strictly follow user request
             # For SVM/MLP, if user didn't provide grid, we skip or use default? 
             # User didn't provide SVM/MLP in the specific request list, so I'll leave them out of the explicit map.
             pass

        if model_type not in self.PARAM_GRIDS:
             raise ValueError(f"No hyperparameter grid defined for {model_type}")

        # 1. Get Base Model
        # We instantiate a fresh base model. 
        # Note: RetrainingService registry stores classes. using that would be good consistency.
        base_model_class = RetrainingServiceWithScaler.MODEL_REGISTRY[model_type]["class"]
        # Use default params as base, but tuning will override
        default_params = RetrainingServiceWithScaler.MODEL_REGISTRY[model_type]["params"]
        
        # Filter params that might conflict or we want to fix (e.g. random_state)
        fixed_params = {k: v for k, v in default_params.items() if k in ['random_state', 'n_jobs', 'class_weight']}
        model = base_model_class(**fixed_params)

        # 2. Create Pipeline
        # Pipeline ensures scaling happens within CV folds
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('classifier', model)
        ])

        # 3. Get Grid
        param_grid = self.PARAM_GRIDS[model_type]
        
        logger.info(f"🔧 Tuning {model_type} with {self.cv}-fold CV...")
        logger.info(f"   Grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")

        # 4. Search
        # User requested RandomizedSearchCV
        n_iter = 50 # Limit iterations for efficiency
        
        # Calculate total combinations to avoid setting n_iter > total
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        if n_iter > total_combinations:
            n_iter = total_combinations
            
        logger.info(f"   Running RandomizedSearchCV with {n_iter} iterations...")
        
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42),
            n_jobs=self.n_jobs,
            verbose=1,
            random_state=42
        )
        
        search.fit(X, y)
        
        logger.info(f"✓ Tuning complete. Best {scoring}: {search.best_score_:.4f}")
        logger.info(f"   Best params: {search.best_params_}")
        
        return {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "best_estimator": search.best_estimator_, # This is the full pipeline
            "cv_results": search.cv_results_
        }

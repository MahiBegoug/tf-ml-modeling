"""
Feature Schema Utilities

Utilities for loading and working with feature schemas.
Feature schemas define the order and names of features expected by models.
"""

import os
import pandas as pd
from typing import List, Optional
from ml.utils.logger import logger


DEFAULT_SCHEMAS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "features")

def load_feature_schema(schema_name: str, schemas_dir: str = None) -> List[str]:
    """
    Load feature names from a schema CSV file.
    
    Args:
        schema_name: Name of the schema file
        schemas_dir: Directory containing schema files (default: package features dir)
    """
    if schemas_dir is None:
        schemas_dir = DEFAULT_SCHEMAS_DIR
        
    schema_path = os.path.join(schemas_dir, schema_name)
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(
            f"Feature schema not found: {schema_path}\n"
            f"Available schemas: {list_available_schemas(schemas_dir)}"
        )
    
    try:
        df = pd.read_csv(schema_path)
        
        if 'Feature' not in df.columns:
            raise ValueError(
                f"Schema file must have a 'Feature' column. "
                f"Found columns: {list(df.columns)}"
            )
        
        # Extract feature names (remove any empty rows)
        features = df['Feature'].dropna().tolist()
        features = [f.strip() for f in features if f.strip()]
        
        if len(features) == 0:
            raise ValueError(f"Schema file is empty: {schema_path}")
        
        logger.info(f"✓ Loaded feature schema: {schema_name} ({len(features)} features)")
        return features
        
    except Exception as e:
        logger.error(f"Failed to load feature schema: {e}")
        raise


def list_available_schemas(schemas_dir: str = "feature_schemas") -> List[str]:
    """
    List all available feature schema files.
    
    Args:
        schemas_dir: Directory containing schema files
        
    Returns:
        List of schema filenames
    """
    if not os.path.exists(schemas_dir):
        return []
    
    return [
        f for f in os.listdir(schemas_dir)
        if f.endswith('.csv')
    ]


def get_schema_for_model_type(model_type: str, schemas_dir: str = "feature_schemas") -> Optional[str]:
    """
    Get the default schema filename for a model type.
    
    Args:
        model_type: Type of model (e.g., "lightgbm", "randomforest")
        schemas_dir: Directory containing schema files
        
    Returns:
        Schema filename if found, None otherwise
        
    Example:
        >>> get_schema_for_model_type("lightgbm")
        'lightgbm_features.csv'
    """
    # Try exact match first
    schema_name = f"{model_type}_features.csv"
    schema_path = os.path.join(schemas_dir, schema_name)
    
    if os.path.exists(schema_path):
        return schema_name
    
    # Try common variations
    variations = [
        f"{model_type.lower()}_features.csv",
        f"{model_type.upper()}_features.csv",
        f"{model_type.replace('_', '')}_features.csv",
    ]
    
    for var in variations:
        var_path = os.path.join(schemas_dir, var)
        if os.path.exists(var_path):
            return var
    
    return None


def validate_feature_schema(schema_name: str, schemas_dir: str = "feature_schemas") -> bool:
    """
    Validate that a feature schema file is valid.
    
    Args:
        schema_name: Name of the schema file
        schemas_dir: Directory containing schema files
        
    Returns:
        True if valid, False otherwise
    """
    try:
        features = load_feature_schema(schema_name, schemas_dir)
        return len(features) > 0
    except Exception:
        return False

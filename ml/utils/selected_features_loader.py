from typing import List
import pandas as pd
import os


def load_selected_features(model_name: str) -> List[str]:
    """
    Loads features of the specified model from a CSV file.

    Args:
        model_name (str): Name of the model (e.g., 'randomforest').

    Returns:
        List[str]: Ordered list of features to use.
    """
    path = os.path.join("feature_schemas", f"{model_name}_features.csv")

    try:
        df = pd.read_csv(path)
        if "Feature" not in df.columns:
            raise ValueError(f"The file {path} does not contain a 'Feature' column")
        return df["Feature"].tolist()
    except Exception as e:
        raise RuntimeError(f"Error loading features from {path}: {e}")

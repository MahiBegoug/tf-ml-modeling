"""
Evaluation metrics for defect prediction models.

This module provides metrics specifically designed for imbalanced binary classification,
which is common in software defect prediction where defects are rare.
"""
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)


def calculate_gmean(y_true, y_pred) -> float:
    """
    Calculate G-Mean (Geometric Mean of Sensitivity and Specificity).
    
    G-Mean is particularly useful for imbalanced datasets as it balances
    the performance on both classes. A high G-Mean indicates good performance
    on both the majority and minority classes.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        
    Returns:
        float: G-Mean score (0.0 to 1.0)
        
    References:
        Kubat, M., & Matwin, S. (1997). Addressing the curse of imbalanced 
        training sets: one-sided selection.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle edge cases
    if cm.shape != (2, 2):
        # If only one class is present, return 0
        return 0.0
    
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate sensitivity (recall/TPR) and specificity (TNR)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # G-Mean is the geometric mean of sensitivity and specificity
    gmean = np.sqrt(sensitivity * specificity)
    
    return gmean


def calculate_all_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for defect prediction.
    
    Includes both traditional metrics and imbalance-aware metrics:
    - MCC: Matthews Correlation Coefficient (best for imbalanced data)
    - G-Mean: Geometric Mean of Sensitivity and Specificity
    - Accuracy, Precision, Recall, F1: Traditional metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dict containing all calculated metrics
        
    Example:
        >>> metrics = calculate_all_metrics(y_true, y_pred)
        >>> print(f"MCC: {metrics['mcc']:.3f}")
        >>> print(f"G-Mean: {metrics['gmean']:.3f}")
    """
    metrics = {
        # Imbalance-aware metrics (primary)
        "mcc": matthews_corrcoef(y_true, y_pred),
        "gmean": calculate_gmean(y_true, y_pred),
        
        # Traditional metrics (secondary)
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    return metrics


def calculate_confusion_matrix_metrics(y_true, y_pred) -> Dict[str, Any]:
    """
    Calculate detailed confusion matrix metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dict containing:
        - tn, fp, fn, tp: Confusion matrix values
        - sensitivity: True Positive Rate (Recall)
        - specificity: True Negative Rate
        - fpr: False Positive Rate
        - fnr: False Negative Rate
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape != (2, 2):
        return {
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
            "sensitivity": 0.0, "specificity": 0.0,
            "fpr": 0.0, "fnr": 0.0
        }
    
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0          # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0          # False Negative Rate
    
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr
    }


def format_metrics_summary(metrics: Dict[str, float], model_name: str = None) -> str:
    """
    Format metrics into a human-readable summary string.
    
    Args:
        metrics: Dictionary of metrics from calculate_all_metrics()
        model_name: Optional model name to include in summary
        
    Returns:
        Formatted string summary
    """
    lines = []
    
    if model_name:
        lines.append(f"Model: {model_name}")
        lines.append("-" * 50)
    
    lines.append("Primary Metrics (Imbalance-Aware):")
    lines.append(f"  MCC:    {metrics.get('mcc', 0):.3f}")
    lines.append(f"  G-Mean: {metrics.get('gmean', 0):.3f}")
    
    lines.append("\nTraditional Metrics:")
    lines.append(f"  Accuracy:  {metrics.get('accuracy', 0):.3f}")
    lines.append(f"  Precision: {metrics.get('precision', 0):.3f}")
    lines.append(f"  Recall:    {metrics.get('recall', 0):.3f}")
    lines.append(f"  F1-Score:  {metrics.get('f1', 0):.3f}")
    
    return "\n".join(lines)


def compare_models(results: Dict[str, Dict[str, float]], 
                   sort_by: str = "mcc") -> List[tuple]:
    """
    Compare multiple models and rank them by a specified metric.
    
    Args:
        results: Dict mapping model_name to metrics dict
        sort_by: Metric to sort by (default: 'mcc')
        
    Returns:
        List of (model_name, metrics) tuples sorted by the specified metric
        
    Example:
        >>> results = {
        ...     "rf": {"mcc": 0.85, "gmean": 0.82},
        ...     "lr": {"mcc": 0.65, "gmean": 0.60}
        ... }
        >>> ranked = compare_models(results, sort_by="mcc")
        >>> print(f"Best model: {ranked[0][0]}")
    """
    # Filter out models with errors
    valid_results = {
        name: metrics 
        for name, metrics in results.items() 
        if "error" not in metrics and sort_by in metrics
    }
    
    # Sort by the specified metric (descending)
    ranked = sorted(
        valid_results.items(),
        key=lambda x: x[1][sort_by],
        reverse=True
    )
    
    return ranked

"""
Model evaluation utilities.

Provides comprehensive evaluation for imbalanced binary classification:
- Probability-based metrics (PR-AUC, ROC-AUC)
- Threshold-based metrics (precision, recall, confusion matrix)
- Operational metrics (precision@K, false positive rate)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss
)


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for binary classification.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        threshold: Decision threshold (if None, only probability metrics returned)
    
    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    
    # Probability-based metrics (threshold-independent)
    metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    
    # Log loss (calibration quality)
    metrics["log_loss"] = log_loss(y_true, y_proba)
    
    # Threshold-based metrics
    if threshold is not None:
        y_pred = (y_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (
            metrics["precision"] + metrics["recall"]
        ) if (metrics["precision"] + metrics["recall"]) > 0 else 0.0
        
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Counts
        metrics["true_positives"] = int(tp)
        metrics["false_positives"] = int(fp)
        metrics["true_negatives"] = int(tn)
        metrics["false_negatives"] = int(fn)
    
    return metrics


def compute_precision_at_k(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    k: int = 1000
) -> float:
    """
    Compute precision in the top K predictions.
    
    This is a key operational metric:
    "Of the top K transactions we flag, how many are actually fraud?"
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        k: Number of top predictions to consider
    
    Returns:
        Precision at K
    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    
    # Get indices of top K predictions
    k = min(k, len(y_proba))
    top_k_indices = np.argsort(y_proba)[-k:][::-1]
    
    # Compute precision in top K
    return y_true[top_k_indices].mean()


def evaluate_model(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    split_name: str = "Test",
    print_report: bool = True
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with formatted output.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        threshold: Decision threshold
        split_name: Name for logging (e.g., "Test", "Validation")
        print_report: Whether to print detailed report
    
    Returns:
        Dictionary of all metrics
    """
    metrics = compute_metrics(y_true, y_proba, threshold)
    
    # Add precision at K metrics
    for k in [100, 500, 1000]:
        if k <= len(y_true):
            metrics[f"precision_at_{k}"] = compute_precision_at_k(y_true, y_proba, k)
    
    if print_report:
        print(f"\n{'=' * 60}")
        print(f"{split_name} Set Evaluation (Threshold: {threshold:.4f})")
        print(f"{'=' * 60}")
        
        print(f"\nProbability Metrics:")
        print(f"  PR-AUC:   {metrics['pr_auc']:.4f}")
        print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
        print(f"  Log Loss: {metrics['log_loss']:.4f}")
        
        print(f"\nThreshold Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics['true_negatives']:,}  FP: {metrics['false_positives']:,}")
        print(f"  FN: {metrics['false_negatives']:,}  TP: {metrics['true_positives']:,}")
        
        print(f"\nOperational Metrics:")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
        
        y_pred = (np.array(y_proba) >= threshold).astype(int)
        flagged_count = y_pred.sum()
        flagged_pct = flagged_count / len(y_pred)
        print(f"  Transactions Flagged: {flagged_count:,} ({flagged_pct:.2%})")
        
        # Precision at K
        print(f"\nPrecision @ K (Review Queue Quality):")
        for k in [100, 500, 1000]:
            key = f"precision_at_{k}"
            if key in metrics:
                print(f"  Precision @ {k}: {metrics[key]:.4f}")
    
    return metrics


def get_classification_report(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float
) -> str:
    """
    Get sklearn classification report string.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        threshold: Decision threshold
    
    Returns:
        Classification report string
    """
    y_pred = (np.array(y_proba) >= threshold).astype(int)
    return classification_report(
        y_true, y_pred,
        target_names=["Legitimate", "Fraud"]
    )

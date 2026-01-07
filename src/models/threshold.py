"""
Threshold selection for fraud detection.

Key insight: The default 0.5 threshold is rarely optimal for imbalanced data.
We select thresholds based on operational constraints:
- Target precision (control false positive rate)
- Target recall (control fraud catch rate)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.metrics import precision_recall_curve

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import get_settings


def select_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    target_precision: Optional[float] = None,
    target_recall: Optional[float] = None,
    settings: Optional[object] = None
) -> Tuple[float, dict]:
    """
    Select decision threshold based on operational constraints.
    
    Since false positives are costly in this system, we typically
    select based on target precision (high precision = low FP rate).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        target_precision: Minimum precision required (default from settings)
        target_recall: Minimum recall required (alternative to precision)
        settings: Configuration settings
    
    Returns:
        Tuple of (selected_threshold, achieved_metrics)
    """
    if settings is None:
        settings = get_settings()
    
    if target_precision is None and target_recall is None:
        target_precision = settings.threshold_config.target_precision
    
    # Compute precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Handle edge case: last element of precisions/recalls has no threshold
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    
    selected_threshold = settings.threshold_config.default_threshold
    achieved_precision = 0.0
    achieved_recall = 0.0
    
    if target_precision is not None:
        # Find threshold that achieves target precision with highest recall
        valid_mask = precisions >= target_precision
        
        if valid_mask.any():
            # Among valid thresholds, pick the one with highest recall
            valid_indices = np.where(valid_mask)[0]
            best_idx = valid_indices[np.argmax(recalls[valid_indices])]
            
            selected_threshold = thresholds[best_idx]
            achieved_precision = precisions[best_idx]
            achieved_recall = recalls[best_idx]
        else:
            # Cannot achieve target precision, use highest available
            best_idx = np.argmax(precisions)
            selected_threshold = thresholds[best_idx]
            achieved_precision = precisions[best_idx]
            achieved_recall = recalls[best_idx]
            
            print(f"WARNING: Cannot achieve target precision {target_precision:.0%}")
            print(f"         Maximum achievable: {achieved_precision:.2%}")
    
    elif target_recall is not None:
        # Find threshold that achieves target recall with highest precision
        valid_mask = recalls >= target_recall
        
        if valid_mask.any():
            valid_indices = np.where(valid_mask)[0]
            best_idx = valid_indices[np.argmax(precisions[valid_indices])]
            
            selected_threshold = thresholds[best_idx]
            achieved_precision = precisions[best_idx]
            achieved_recall = recalls[best_idx]
        else:
            # Cannot achieve target recall, use highest available
            best_idx = np.argmax(recalls)
            selected_threshold = thresholds[best_idx]
            achieved_precision = precisions[best_idx]
            achieved_recall = recalls[best_idx]
            
            print(f"WARNING: Cannot achieve target recall {target_recall:.0%}")
            print(f"         Maximum achievable: {achieved_recall:.2%}")
    
    print(f"\nThreshold Selection:")
    print(f"  Target Precision: {target_precision:.0%}" if target_precision else "")
    print(f"  Selected Threshold: {selected_threshold:.4f}")
    print(f"  Achieved Precision: {achieved_precision:.2%}")
    print(f"  Achieved Recall: {achieved_recall:.2%}")
    
    return selected_threshold, {
        "threshold": selected_threshold,
        "precision": achieved_precision,
        "recall": achieved_recall,
        "target_precision": target_precision,
        "target_recall": target_recall
    }


def get_threshold_analysis(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[list] = None
) -> pd.DataFrame:
    """
    Analyze performance at multiple thresholds.
    
    Useful for understanding precision-recall tradeoffs.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        thresholds: Specific thresholds to analyze (default: 0.1 to 0.9)
    
    Returns:
        DataFrame with metrics at each threshold
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    from .evaluation import compute_metrics
    
    results = []
    for t in thresholds:
        metrics = compute_metrics(y_true, y_proba, threshold=t)
        metrics["threshold"] = t
        results.append(metrics)
    
    return pd.DataFrame(results)


def plot_precision_recall_curve(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    selected_threshold: Optional[float] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot precision-recall curve with optional threshold marker.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        selected_threshold: Threshold to highlight on curve
        save_path: Path to save the plot (if None, displays interactively)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Main curve
    ax.plot(recalls, precisions, "b-", linewidth=2, label="Precision-Recall Curve")
    
    # Mark selected threshold
    if selected_threshold is not None:
        # Find closest threshold
        idx = np.argmin(np.abs(thresholds - selected_threshold))
        ax.scatter(
            [recalls[idx]], [precisions[idx]],
            s=100, c="red", marker="o", zorder=5,
            label=f"Selected (t={selected_threshold:.2f})"
        )
    
    # Add iso-F1 curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        y[y < 0] = np.nan
        ax.plot(x, y, "--", color="gray", alpha=0.3)
        ax.annotate(f"F1={f_score:.1f}", xy=(0.9, y[np.isnan(y) == False][-1]))
    
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved precision-recall curve to {save_path}")
    else:
        plt.show()
    
    plt.close()

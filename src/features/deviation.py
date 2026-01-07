"""
Deviation features for anomaly detection.

These features measure how unusual the current transaction is
compared to user's historical behavior and merchant norms.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def compute_deviation_features(
    df: pd.DataFrame,
    merchant_medians: Optional[Dict[str, float]] = None,
    fit_merchant_stats: bool = False
) -> Tuple[pd.DataFrame, dict]:
    """
    Compute deviation/anomaly signal features (vectorized for performance).
    
    Features:
    - amount_zscore_user: (amount - user_mean) / user_std
    - amount_vs_user_median_ratio: amount / user_median
    - time_since_last_transaction: hours since previous transaction
    - amount_percentile_user: percentile rank in user's history
    - amount_vs_merchant_median: amount / merchant_median
    
    Args:
        df: Transaction DataFrame (MUST be sorted by timestamp)
        merchant_medians: Pre-computed merchant median amounts
        fit_merchant_stats: Whether to compute merchant stats from this data
    
    Returns:
        Tuple of (DataFrame with features, dict with merchant stats)
    """
    df = df.copy()
    stats_dict = {}
    
    # Sort by user and timestamp for proper rolling calculations
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    
    # Compute merchant medians if needed
    if fit_merchant_stats or merchant_medians is None:
        merchant_medians = df.groupby("merchant_category")["amount"].median().to_dict()
    stats_dict["merchant_medians"] = merchant_medians
    
    # Global median as fallback
    global_median = df["amount"].median()
    stats_dict["global_median"] = global_median
    
    # ================================================================
    # TIME SINCE LAST TRANSACTION (vectorized)
    # ================================================================
    # Calculate time difference from previous transaction per user
    df["time_since_last_transaction"] = df.groupby("user_id")["timestamp"].diff()
    df["time_since_last_transaction"] = df["time_since_last_transaction"].dt.total_seconds() / 3600
    
    # Fill first transaction for each user with a large value
    max_time = df["time_since_last_transaction"].max()
    default_time = max_time if pd.notna(max_time) else 24 * 30  # 30 days default
    df["time_since_last_transaction"] = df["time_since_last_transaction"].fillna(default_time)
    
    # ================================================================
    # AMOUNT VS MERCHANT MEDIAN (vectorized)
    # ================================================================
    df["merchant_median_lookup"] = df["merchant_category"].map(merchant_medians)
    df["merchant_median_lookup"] = df["merchant_median_lookup"].fillna(global_median)
    df["amount_vs_merchant_median"] = df["amount"] / df["merchant_median_lookup"].replace(0, 1)
    df = df.drop(columns=["merchant_median_lookup"])
    
    # ================================================================
    # USER-BASED FEATURES (expanding window, shifted to exclude current)
    # ================================================================
    # These require looking at past transactions only
    
    # Rolling mean (shifted to exclude current row)
    df["user_rolling_mean"] = df.groupby("user_id")["amount"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    
    # Rolling std (shifted, needs at least 2 values)
    df["user_rolling_std"] = df.groupby("user_id")["amount"].transform(
        lambda x: x.shift(1).expanding(min_periods=2).std()
    )
    
    # Rolling median (shifted)
    df["user_rolling_median"] = df.groupby("user_id")["amount"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).median()
    )
    
    # Z-score
    df["amount_zscore_user"] = (df["amount"] - df["user_rolling_mean"]) / df["user_rolling_std"].replace(0, 1)
    df["amount_zscore_user"] = df["amount_zscore_user"].fillna(0)
    
    # Ratio to user median
    df["amount_vs_user_median_ratio"] = df["amount"] / df["user_rolling_median"].replace(0, 1)
    df["amount_vs_user_median_ratio"] = df["amount_vs_user_median_ratio"].fillna(1)
    
    # ================================================================
    # PERCENTILE RANK (approximate using rank within expanding window)
    # ================================================================
    # This is an approximation - for each row, we compute what percentile
    # the current amount would be in the user's history
    
    def compute_percentile_rank(group):
        """Compute percentile rank for each transaction within user history."""
        result = pd.Series(index=group.index, dtype=float)
        amounts = group["amount"].values
        
        for i in range(len(amounts)):
            if i == 0:
                result.iloc[i] = 0.5  # No history
            else:
                past_amounts = amounts[:i]
                current = amounts[i]
                result.iloc[i] = (past_amounts < current).mean()
        
        return result
    
    # For large datasets, use an approximation based on rolling rank
    if len(df) > 50000:
        # Approximation: use cumulative count below / cumulative count
        df["amount_percentile_user"] = df.groupby("user_id")["amount"].transform(
            lambda x: x.shift(1).expanding().rank(pct=True).shift(-1).fillna(0.5)
        )
    else:
        # Exact computation for smaller datasets
        # Exact computation for smaller datasets
        # Fix: ensure result aligns with original index
        groups = df.groupby("user_id")
        results = []
        for name, group in groups:
            res = compute_percentile_rank(group)
            results.append(res)
        
        if results:
            df["amount_percentile_user"] = pd.concat(results)
        else:
            df["amount_percentile_user"] = 0.5
    
    df["amount_percentile_user"] = df["amount_percentile_user"].fillna(0.5)
    
    # ================================================================
    # CLEANUP
    # ================================================================
    # Drop intermediate columns
    df = df.drop(columns=["user_rolling_mean", "user_rolling_std", "user_rolling_median"], errors="ignore")
    
    # Clip extreme values for stability
    df["amount_zscore_user"] = df["amount_zscore_user"].clip(-10, 10)
    df["amount_vs_user_median_ratio"] = df["amount_vs_user_median_ratio"].clip(0, 100)
    df["amount_vs_merchant_median"] = df["amount_vs_merchant_median"].clip(0, 100)
    
    # Re-sort by original timestamp order
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df, stats_dict

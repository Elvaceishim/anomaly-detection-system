"""
User behavioral aggregate features.

These features capture a user's historical transaction patterns
using ONLY past transactions (timestamp < current_transaction.timestamp).

All window-based computations exclude the current transaction to prevent leakage.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import get_settings


def compute_user_aggregates(
    df: pd.DataFrame,
    settings: Optional[object] = None
) -> pd.DataFrame:
    """
    Compute user behavioral aggregate features.
    
    Features:
    - mean_amount_last_7d: Average transaction amount in past 7 days
    - std_amount_last_30d: Standard deviation of amounts in past 30 days
    - transaction_count_last_24h: Number of transactions in past 24 hours
    - total_amount_last_7d: Sum of transaction amounts in past 7 days
    - unique_merchants_last_30d: Count of distinct merchants in past 30 days
    - unique_locations_last_30d: Count of distinct locations in past 30 days
    
    Args:
        df: Transaction DataFrame (MUST be sorted by timestamp)
        settings: Configuration settings (uses defaults if None)
    
    Returns:
        DataFrame with additional aggregate columns
    """
    if settings is None:
        settings = get_settings()
    
    df = df.copy()
    
    # Initialize feature columns
    df["mean_amount_last_7d"] = np.nan
    df["std_amount_last_30d"] = np.nan
    df["transaction_count_last_24h"] = 0
    df["total_amount_last_7d"] = 0.0
    df["unique_merchants_last_30d"] = 0
    df["unique_locations_last_30d"] = 0
    
    # Pre-compute for efficiency: group by user
    # This is more efficient than row-by-row iteration for large datasets
    
    # Convert to hours for window calculations
    hours_24 = 24
    days_7_hours = settings.feature_windows.short_term_days * 24
    days_30_hours = settings.feature_windows.medium_term_days * 24
    
    # Process each user's transactions
    for user_id in df["user_id"].unique():
        user_mask = df["user_id"] == user_id
        user_df = df[user_mask].copy()
        
        # Iterate through user's transactions chronologically
        for i, (idx, row) in enumerate(user_df.iterrows()):
            current_time = row["timestamp"]
            
            # Get all transactions BEFORE this one
            past_mask = user_df["timestamp"] < current_time
            past_df = user_df[past_mask]
            
            if len(past_df) == 0:
                # No history for this user yet
                continue
            
            # Time-window masks
            time_diffs = (current_time - past_df["timestamp"]).dt.total_seconds() / 3600
            
            mask_24h = time_diffs <= hours_24
            mask_7d = time_diffs <= days_7_hours
            mask_30d = time_diffs <= days_30_hours
            
            # Compute aggregates
            
            # mean_amount_last_7d
            past_7d = past_df[mask_7d]
            if len(past_7d) > 0:
                df.loc[idx, "mean_amount_last_7d"] = past_7d["amount"].mean()
                df.loc[idx, "total_amount_last_7d"] = past_7d["amount"].sum()
            
            # std_amount_last_30d
            past_30d = past_df[mask_30d]
            if len(past_30d) >= 2:  # Need at least 2 for std
                df.loc[idx, "std_amount_last_30d"] = past_30d["amount"].std()
            
            # transaction_count_last_24h
            df.loc[idx, "transaction_count_last_24h"] = mask_24h.sum()
            
            # unique_merchants_last_30d
            if len(past_30d) > 0:
                df.loc[idx, "unique_merchants_last_30d"] = past_30d["merchant_category"].nunique()
            
            # unique_locations_last_30d
            if len(past_30d) > 0:
                df.loc[idx, "unique_locations_last_30d"] = past_30d["location"].nunique()
    
    return df


def compute_user_aggregates_vectorized(
    df: pd.DataFrame,
    settings: Optional[object] = None
) -> pd.DataFrame:
    """
    Vectorized version for large datasets.
    
    Uses rolling windows with groupby for better performance.
    Note: This is an approximation as rolling windows use fixed periods,
    not exact time deltas. For production, consider using a proper
    time-series library.
    
    Args:
        df: Transaction DataFrame (MUST be sorted by timestamp)
        settings: Configuration settings
    
    Returns:
        DataFrame with additional aggregate columns
    """
    if settings is None:
        settings = get_settings()
    
    df = df.copy()
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    
    # Set timestamp as index for time-based operations
    df = df.set_index("timestamp")
    
    # Group by user and apply rolling windows
    grouped = df.groupby("user_id")
    
    # Rolling 7-day mean amount (use lowercase 'd' to avoid deprecation)
    df["mean_amount_last_7d"] = grouped["amount"].apply(
        lambda x: x.shift(1).rolling("7d", min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    
    # Rolling 30-day std amount
    df["std_amount_last_30d"] = grouped["amount"].apply(
        lambda x: x.shift(1).rolling("30d", min_periods=2).std()
    ).reset_index(level=0, drop=True)
    
    # Rolling 24-hour count (use lowercase 'h')
    df["transaction_count_last_24h"] = grouped["amount"].apply(
        lambda x: x.shift(1).rolling("24h", min_periods=0).count()
    ).reset_index(level=0, drop=True)
    
    # Rolling 7-day sum
    df["total_amount_last_7d"] = grouped["amount"].apply(
        lambda x: x.shift(1).rolling("7d", min_periods=1).sum()
    ).reset_index(level=0, drop=True)
    
    # Reset index for unique counts calculation
    df = df.reset_index()
    
    # Unique merchants last 30d & unique locations last 30d
    # Use cumulative unique count approach: for each row, count how many unique values
    # have appeared for that user up to but not including the current row
    
    # For unique merchants: mark first occurrence per (user, merchant) combo with 1, then cumsum
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["_is_first_merchant"] = (~df.duplicated(subset=["user_id", "merchant_category"])).astype(int)
    df["unique_merchants_last_30d"] = df.groupby("user_id")["_is_first_merchant"].cumsum().shift(1, fill_value=0)
    df["unique_merchants_last_30d"] = df.groupby("user_id")["unique_merchants_last_30d"].transform(
        lambda x: x.fillna(method="ffill").fillna(0)
    ).astype(int)
    
    # For unique locations: same approach
    df["_is_first_location"] = (~df.duplicated(subset=["user_id", "location"])).astype(int)
    df["unique_locations_last_30d"] = df.groupby("user_id")["_is_first_location"].cumsum().shift(1, fill_value=0)
    df["unique_locations_last_30d"] = df.groupby("user_id")["unique_locations_last_30d"].transform(
        lambda x: x.fillna(method="ffill").fillna(0)
    ).astype(int)
    
    # Clean up temporary columns
    df = df.drop(columns=["_is_first_merchant", "_is_first_location"])
    
    # Fill NaN with defaults
    df["mean_amount_last_7d"] = df["mean_amount_last_7d"].fillna(0)
    df["std_amount_last_30d"] = df["std_amount_last_30d"].fillna(0)
    df["transaction_count_last_24h"] = df["transaction_count_last_24h"].fillna(0).astype(int)
    df["total_amount_last_7d"] = df["total_amount_last_7d"].fillna(0)
    
    # Re-sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df

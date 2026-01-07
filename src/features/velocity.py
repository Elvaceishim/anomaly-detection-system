"""
Velocity and burst features for short-term risk detection.

These features capture rapid transaction patterns that may indicate:
- Account takeover (attacker draining funds quickly)
- Card testing (multiple small transactions)
- Velocity abuse
"""

import pandas as pd
import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import get_settings


def compute_velocity_features(
    df: pd.DataFrame,
    settings: Optional[object] = None
) -> pd.DataFrame:
    """
    Compute velocity and burst features.
    
    Features:
    - transactions_last_1h: Count of user's transactions in past hour
    - amount_sum_last_1h: Total transaction amount in past hour
    - failed_transaction_count_last_24h: Failed transactions in past 24 hours
    
    Args:
        df: Transaction DataFrame (MUST be sorted by timestamp)
        settings: Configuration settings
    
    Returns:
        DataFrame with additional velocity columns
    """
    if settings is None:
        settings = get_settings()
    
    df = df.copy()
    
    # Initialize columns
    df["transactions_last_1h"] = 0
    df["amount_sum_last_1h"] = 0.0
    df["failed_transaction_count_last_24h"] = 0
    
    # Process each user
    for user_id in df["user_id"].unique():
        user_mask = df["user_id"] == user_id
        user_df = df[user_mask]
        user_indices = user_df.index.tolist()
        
        for idx in user_indices:
            current_time = df.loc[idx, "timestamp"]
            
            # Get past transactions for this user
            past_mask = (df["user_id"] == user_id) & (df["timestamp"] < current_time)
            past_df = df[past_mask]
            
            if len(past_df) == 0:
                continue
            
            # Time differences in hours
            time_diffs = (current_time - past_df["timestamp"]).dt.total_seconds() / 3600
            
            # Transactions in last 1 hour
            mask_1h = time_diffs <= 1
            df.loc[idx, "transactions_last_1h"] = mask_1h.sum()
            df.loc[idx, "amount_sum_last_1h"] = past_df.loc[mask_1h, "amount"].sum()
            
            # Failed transactions in last 24 hours
            mask_24h = time_diffs <= 24
            if "is_failed" in past_df.columns:
                df.loc[idx, "failed_transaction_count_last_24h"] = (
                    past_df.loc[mask_24h, "is_failed"].sum()
                )
    
    return df


def compute_velocity_features_vectorized(
    df: pd.DataFrame,
    settings: Optional[object] = None
) -> pd.DataFrame:
    """
    Vectorized velocity feature computation for large datasets.
    
    Uses rolling windows with time-based indexing.
    
    Args:
        df: Transaction DataFrame (MUST be sorted by timestamp)
        settings: Configuration settings
    
    Returns:
        DataFrame with additional velocity columns
    """
    if settings is None:
        settings = get_settings()
    
    df = df.copy()
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    
    # Set timestamp as index
    df_indexed = df.set_index("timestamp")
    
    # Group by user
    grouped = df_indexed.groupby("user_id")
    
    # Rolling 1-hour transaction count
    # shift(1) ensures we don't count the current transaction
    df["transactions_last_1h"] = grouped["amount"].apply(
        lambda x: x.shift(1).rolling("1H", min_periods=0).count()
    ).reset_index(level=0, drop=True).fillna(0).astype(int).values
    
    # Rolling 1-hour amount sum
    df["amount_sum_last_1h"] = grouped["amount"].apply(
        lambda x: x.shift(1).rolling("1H", min_periods=0).sum()
    ).reset_index(level=0, drop=True).fillna(0).values
    
    # Rolling 24-hour failed transaction count
    if "is_failed" in df.columns:
        df_indexed["is_failed_int"] = df_indexed["is_failed"].astype(int)
        df["failed_transaction_count_last_24h"] = df_indexed.groupby("user_id")["is_failed_int"].apply(
            lambda x: x.shift(1).rolling("24H", min_periods=0).sum()
        ).reset_index(level=0, drop=True).fillna(0).astype(int).values
    else:
        df["failed_transaction_count_last_24h"] = 0
    
    return df

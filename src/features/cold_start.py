"""
Cold-start and user stability features.

These features help the model handle new users appropriately:
- New users have limited history, making behavior-based features unreliable
- The model can learn to weight features differently for new users
"""

import pandas as pd
import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import get_settings


def compute_cold_start_features(
    df: pd.DataFrame,
    settings: Optional[object] = None
) -> pd.DataFrame:
    """
    Compute cold-start and user stability features.
    
    Features:
    - user_transaction_count_to_date: Total transactions for user before current
    - is_new_user: Binary flag if user has < threshold transactions
    
    Args:
        df: Transaction DataFrame (MUST be sorted by timestamp)
        settings: Configuration settings
    
    Returns:
        DataFrame with additional cold-start columns
    """
    if settings is None:
        settings = get_settings()
    
    df = df.copy()
    
    # Get threshold from config
    new_user_threshold = settings.cold_start_config.new_user_threshold
    
    # Initialize columns
    df["user_transaction_count_to_date"] = 0
    df["is_new_user"] = 1  # Default to new user
    
    # Track transaction count per user as we iterate
    user_counts = {}
    
    for idx in df.index:
        user_id = df.loc[idx, "user_id"]
        
        # Get current count (before this transaction)
        current_count = user_counts.get(user_id, 0)
        
        # Set features
        df.loc[idx, "user_transaction_count_to_date"] = current_count
        df.loc[idx, "is_new_user"] = 1 if current_count < new_user_threshold else 0
        
        # Increment count for next transaction
        user_counts[user_id] = current_count + 1
    
    return df


def compute_cold_start_features_vectorized(
    df: pd.DataFrame,
    settings: Optional[object] = None
) -> pd.DataFrame:
    """
    Vectorized cold-start feature computation.
    
    Args:
        df: Transaction DataFrame (MUST be sorted by timestamp)
        settings: Configuration settings
    
    Returns:
        DataFrame with additional cold-start columns
    """
    if settings is None:
        settings = get_settings()
    
    df = df.copy()
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    
    new_user_threshold = settings.cold_start_config.new_user_threshold
    
    # Cumulative count per user (shifted to exclude current transaction)
    df["user_transaction_count_to_date"] = df.groupby("user_id").cumcount()
    
    # Is new user
    df["is_new_user"] = (df["user_transaction_count_to_date"] < new_user_threshold).astype(int)
    
    return df


def get_user_account_age(
    df: pd.DataFrame,
    user_first_seen: Optional[dict] = None
) -> pd.Series:
    """
    Compute account age in days for each transaction.
    
    Args:
        df: Transaction DataFrame
        user_first_seen: Dict mapping user_id to first transaction date
    
    Returns:
        Series with account age in days
    """
    if user_first_seen is None:
        user_first_seen = df.groupby("user_id")["timestamp"].min().to_dict()
    
    def get_age(row):
        first_seen = user_first_seen.get(row["user_id"])
        if first_seen is None:
            return 0
        return (row["timestamp"] - first_seen).days
    
    return df.apply(get_age, axis=1)

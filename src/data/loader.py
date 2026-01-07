"""
Data loading utilities with temporal splitting.

Key principle: All splits are temporal to prevent data leakage.
Never use random splits for time-series fraud detection.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime


def load_transactions(
    filepath: str | Path,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load transaction data from CSV or parquet file.
    
    Args:
        filepath: Path to the data file
        parse_dates: Whether to parse timestamp column as datetime
    
    Returns:
        DataFrame with transaction data
    """
    filepath = Path(filepath)
    
    if filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif filepath.suffix == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    if parse_dates and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Sort by timestamp for temporal operations
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df


def temporal_split(
    df: pd.DataFrame,
    train_end: str | datetime,
    val_end: str | datetime,
    timestamp_col: str = "timestamp"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally into train/validation/test sets.
    
    This is the ONLY correct way to split time-series data for fraud detection.
    Random splits leak future information into training, leading to:
    - Overly optimistic validation metrics
    - Poor production performance
    
    Args:
        df: DataFrame with transactions
        train_end: End date for training set (exclusive)
        val_end: End date for validation set (exclusive)
        timestamp_col: Name of the timestamp column
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    
    Example:
        >>> train_df, val_df, test_df = temporal_split(
        ...     df,
        ...     train_end="2025-09-01",
        ...     val_end="2025-11-01"
        ... )
    """
    # Ensure data is sorted
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Convert string dates to datetime if needed
    if isinstance(train_end, str):
        train_end = pd.to_datetime(train_end)
    if isinstance(val_end, str):
        val_end = pd.to_datetime(val_end)
    
    # Split by timestamp
    train_df = df[df[timestamp_col] < train_end].copy()
    val_df = df[(df[timestamp_col] >= train_end) & (df[timestamp_col] < val_end)].copy()
    test_df = df[df[timestamp_col] >= val_end].copy()
    
    # Log split statistics
    total = len(df)
    print(f"Temporal Split Statistics:")
    print(f"  Train: {len(train_df):,} ({len(train_df)/total:.1%}) | "
          f"< {train_end.strftime('%Y-%m-%d')}")
    print(f"  Val:   {len(val_df):,} ({len(val_df)/total:.1%}) | "
          f"{train_end.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/total:.1%}) | "
          f">= {val_end.strftime('%Y-%m-%d')}")
    
    # Log fraud rates if label exists
    if "is_fraud" in df.columns:
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            fraud_rate = split_df["is_fraud"].mean()
            print(f"  {name} fraud rate: {fraud_rate:.2%}")
    
    return train_df, val_df, test_df


def create_user_history_lookup(
    df: pd.DataFrame,
    max_history_days: int = 90
) -> dict:
    """
    Create a lookup structure for efficient user history queries.
    
    This pre-computes user transaction histories to speed up feature computation.
    
    Args:
        df: Full transaction DataFrame (sorted by timestamp)
        max_history_days: Maximum days of history to keep per user
    
    Returns:
        Dictionary mapping user_id to their transaction history DataFrame
    """
    user_histories = {}
    
    for user_id, user_df in df.groupby("user_id"):
        # Keep only the columns needed for feature computation
        history_cols = ["timestamp", "amount", "merchant_category", "location", "is_failed"]
        available_cols = [c for c in history_cols if c in user_df.columns]
        user_histories[user_id] = user_df[available_cols].copy()
    
    return user_histories


def get_merchant_statistics(
    df: pd.DataFrame,
    cutoff_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Compute merchant-level statistics for amount_vs_merchant_median feature.
    
    Args:
        df: Transaction DataFrame
        cutoff_date: Only use transactions before this date (for temporal safety)
    
    Returns:
        DataFrame with merchant statistics (median amount, transaction count)
    """
    if cutoff_date is not None:
        df = df[df["timestamp"] < cutoff_date]
    
    merchant_stats = df.groupby("merchant_category").agg(
        median_amount=("amount", "median"),
        transaction_count=("amount", "count")
    ).reset_index()
    
    return merchant_stats

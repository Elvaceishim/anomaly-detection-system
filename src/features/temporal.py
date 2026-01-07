"""
Temporal features derived from transaction timestamp.

These features capture time-based patterns that may indicate fraud:
- Unusual hours (late night transactions)
- Weekend patterns
- Day of week variations
"""

import pandas as pd
import numpy as np


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal features from transaction timestamp.
    
    Features:
    - hour_of_day: Hour when transaction occurred (0-23)
    - day_of_week: Day index (0=Monday, 6=Sunday)
    - is_weekend: Binary flag for Saturday/Sunday
    - is_night_transaction: Binary flag for late night (12am-5am)
    
    Args:
        df: Transaction DataFrame with 'timestamp' column
    
    Returns:
        DataFrame with additional temporal columns
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Hour of day (0-23)
    df["hour_of_day"] = df["timestamp"].dt.hour
    
    # Day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    # Is weekend (Saturday=5 or Sunday=6)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Is night transaction (12am-5am, i.e., hours 0-4)
    df["is_night_transaction"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] < 5)).astype(int)
    
    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical encoding of time features for models that benefit from it.
    
    Cyclical encoding represents hour and day as sine/cosine pairs,
    which helps models understand that hour 23 is close to hour 0.
    
    Args:
        df: DataFrame with hour_of_day and day_of_week columns
    
    Returns:
        DataFrame with additional cyclical features
    """
    df = df.copy()
    
    # Hour cyclical encoding (24-hour cycle)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    
    # Day of week cyclical encoding (7-day cycle)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    return df

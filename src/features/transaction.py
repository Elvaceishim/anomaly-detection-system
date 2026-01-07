"""
Transaction-level features.

These features are derived directly from the current transaction,
with some requiring historical context (is_new_merchant, is_new_location).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import LabelEncoder


def compute_transaction_features(
    df: pd.DataFrame,
    transaction_type_encoder: Optional[LabelEncoder] = None,
    merchant_category_encoder: Optional[LabelEncoder] = None,
    user_histories: Optional[Dict] = None,
    fit_encoders: bool = False
) -> Tuple[pd.DataFrame, dict]:
    """
    Compute transaction-level features (vectorized for performance).
    
    Features:
    - log_transaction_amount: Log-scaled amount to reduce skew
    - transaction_type_encoded: Encoded transaction type
    - merchant_category_encoded: Encoded merchant category
    - is_new_merchant_for_user: First time transacting with this merchant
    - is_new_location_for_user: First time transacting in this location
    
    Args:
        df: Transaction DataFrame (must be sorted by timestamp)
        transaction_type_encoder: Pre-fitted encoder (None if fitting)
        merchant_category_encoder: Pre-fitted encoder (None if fitting)
        user_histories: Pre-computed user histories for efficiency
        fit_encoders: Whether to fit new encoders
    
    Returns:
        Tuple of (DataFrame with features, dict of fitted encoders)
    """
    df = df.copy()
    encoders = {}
    
    # 1. Log transaction amount
    # Add 1 to handle zero amounts, use natural log
    df["log_transaction_amount"] = np.log1p(df["amount"])
    
    # 2. Transaction type encoding
    if fit_encoders or transaction_type_encoder is None:
        transaction_type_encoder = LabelEncoder()
        transaction_type_encoder.fit(df["transaction_type"].astype(str))
    
    df["transaction_type_encoded"] = _safe_transform_vectorized(
        df["transaction_type"].astype(str),
        transaction_type_encoder
    )
    encoders["transaction_type_encoder"] = transaction_type_encoder
    
    # 3. Merchant category encoding
    if fit_encoders or merchant_category_encoder is None:
        merchant_category_encoder = LabelEncoder()
        merchant_category_encoder.fit(df["merchant_category"].astype(str))
    
    df["merchant_category_encoded"] = _safe_transform_vectorized(
        df["merchant_category"].astype(str),
        merchant_category_encoder
    )
    encoders["merchant_category_encoder"] = merchant_category_encoder
    
    # 4. Is new merchant for user (vectorized)
    # 5. Is new location for user (vectorized)
    df["is_new_merchant_for_user"] = _compute_is_new_field_vectorized(
        df, field="merchant_category"
    )
    df["is_new_location_for_user"] = _compute_is_new_field_vectorized(
        df, field="location"
    )
    
    return df, encoders


def _safe_transform_vectorized(series: pd.Series, encoder: LabelEncoder) -> pd.Series:
    """
    Transform with handling for unseen categories (vectorized).
    
    Unseen categories are encoded as -1.
    """
    # Create a mapping dictionary for fast lookup
    class_to_code = {cls: code for code, cls in enumerate(encoder.classes_)}
    
    # Use map for vectorized transformation
    result = series.map(class_to_code)
    
    # Fill NaN (unseen categories) with -1
    result = result.fillna(-1).astype(int)
    
    return result


def _compute_is_new_field_vectorized(
    df: pd.DataFrame,
    field: str
) -> pd.Series:
    """
    Compute whether a field value is new for each user (vectorized).
    
    IMPORTANT: Uses only transactions BEFORE the current one (no leakage).
    
    This vectorized version uses groupby + transform with cumulative operations.
    
    Args:
        df: DataFrame sorted by timestamp
        field: Column name to check (merchant_category or location)
    
    Returns:
        Series of 0/1 indicating if field value is new for user
    """
    # Sort by user and timestamp to ensure chronological order
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    
    # Create a composite key of user_id and field value
    df["_user_field_key"] = df["user_id"].astype(str) + "_" + df[field].astype(str)
    
    # For each (user, field_value) combination, mark the first occurrence as "new"
    # Use groupby + cumcount to get the occurrence number for each combo
    df["_occurrence_count"] = df.groupby("_user_field_key").cumcount()
    
    # First occurrence (count == 0) means it's new
    is_new = (df["_occurrence_count"] == 0).astype(int)
    
    # Clean up temporary columns
    df.drop(columns=["_user_field_key", "_occurrence_count"], inplace=True)
    
    # Re-sort by timestamp to restore original order
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return is_new

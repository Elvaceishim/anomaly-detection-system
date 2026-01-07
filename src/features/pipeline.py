"""
Feature engineering pipeline orchestrator.

This module coordinates all feature computations and ensures:
1. Proper temporal ordering (no data leakage)
2. Consistent encoding between train and inference
3. Missing value handling for cold-start users
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import get_settings, Settings

from .transaction import compute_transaction_features
from .user_aggregates import compute_user_aggregates, compute_user_aggregates_vectorized
from .deviation import compute_deviation_features
from .temporal import compute_temporal_features
from .velocity import compute_velocity_features, compute_velocity_features_vectorized
from .cold_start import compute_cold_start_features, compute_cold_start_features_vectorized


# Threshold for switching to fast mode (row count)
FAST_MODE_THRESHOLD = 10000


class FeatureEngineer:
    """
    Orchestrates feature computation for training and inference.
    
    Key responsibilities:
    - Coordinate all feature modules
    - Fit encoders on training data
    - Store and apply feature statistics
    - Handle missing values consistently
    
    Usage:
        # Training
        engineer = FeatureEngineer()
        X_train, y_train = engineer.fit_transform(train_df)
        engineer.save("models/feature_engineer.pkl")
        
        # Inference
        engineer = FeatureEngineer.load("models/feature_engineer.pkl")
        X = engineer.transform(new_transactions)
    """
    
    def __init__(self, settings: Optional[Settings] = None, fast_mode: Optional[bool] = None):
        """
        Initialize the feature engineer.
        
        Args:
            settings: Configuration settings (uses defaults if None)
            fast_mode: Use vectorized operations (auto-detects if None)
        """
        self.settings = settings or get_settings()
        self.fast_mode = fast_mode
        
        # Fitted state
        self.is_fitted = False
        self.encoders: Dict = {}
        self.merchant_stats: Dict = {}
        self.fill_values: Dict = {}
        
    def _should_use_fast_mode(self, df: pd.DataFrame) -> bool:
        """Determine if fast mode should be used based on dataset size."""
        if self.fast_mode is not None:
            return self.fast_mode
        return len(df) > FAST_MODE_THRESHOLD
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "is_fraud"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit encoders and transform training data.
        
        IMPORTANT: This must only be called on training data.
        Encoders and statistics are fitted here and stored for inference.
        
        Args:
            df: Training DataFrame with transactions
            target_col: Name of the target column
        
        Returns:
            Tuple of (feature DataFrame, target Series)
        """
        use_fast = self._should_use_fast_mode(df)
        mode_str = "FAST (vectorized)" if use_fast else "STANDARD (row-by-row)"
        print(f"Fitting feature engineer on training data... [{mode_str}]")
        print(f"  Dataset size: {len(df):,} transactions")
        
        # Ensure data is sorted by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # 1. Transaction-level features (fit encoders)
        print("  [1/6] Transaction features...")
        df, encoders = compute_transaction_features(
            df,
            fit_encoders=True
        )
        self.encoders.update(encoders)
        
        # 2. User behavioral aggregates
        print("  [2/6] User aggregates...")
        if use_fast:
            df = compute_user_aggregates_vectorized(df, self.settings)
        else:
            df = compute_user_aggregates(df, self.settings)
        
        # 3. Deviation features (fit merchant stats) - uses vectorized internally
        print("  [3/6] Deviation features...")
        df, merchant_stats = compute_deviation_features(
            df,
            fit_merchant_stats=True
        )
        self.merchant_stats = merchant_stats
        
        # 4. Temporal features (already fast)
        print("  [4/6] Temporal features...")
        df = compute_temporal_features(df)
        
        # 5. Velocity features
        print("  [5/6] Velocity features...")
        if use_fast:
            df = compute_velocity_features_vectorized(df, self.settings)
        else:
            df = compute_velocity_features(df, self.settings)
        
        # 6. Cold-start features
        print("  [6/6] Cold-start features...")
        if use_fast:
            df = compute_cold_start_features_vectorized(df, self.settings)
        else:
            df = compute_cold_start_features(df, self.settings)
        
        # 7. Preprocess IEEE features (encode categoricals)
        df = self._preprocess_ieee_features(df, fit=True)
        
        # 8. Handle missing values and store fill values
        # Filter to only columns that actually exist in the dataframe
        all_feature_cols = self.settings.feature_columns
        feature_cols = [c for c in all_feature_cols if c in df.columns]
        
        # Add new deviation features if they exist
        new_features = ["log_amount_vs_user_median", "log_amount_vs_merchant_median"]
        for feat in new_features:
            if feat in df.columns and feat not in feature_cols:
                feature_cols.append(feat)
        
        # Store the actual feature columns used
        self._fitted_feature_cols = feature_cols
        
        X = df[feature_cols].copy()
        
        # Store median values for imputation (only for numeric columns)
        self.fill_values = X.select_dtypes(include=[np.number]).median().to_dict()
        
        # For non-numeric columns, use mode or a default value
        for col in X.select_dtypes(exclude=[np.number]).columns:
            self.fill_values[col] = X[col].mode()[0] if len(X[col].mode()) > 0 else "unknown"
        
        # Fill missing values
        X = X.fillna(self.fill_values)
        
        # Extract target
        y = df[target_col].astype(int)
        
        self.is_fitted = True
        
        print(f"Feature engineering complete. Shape: {X.shape}")
        print(f"Features: {list(X.columns)}")
        
        return X, y
    
    def transform(
        self,
        df: pd.DataFrame,
        include_target: bool = False,
        target_col: str = "is_fraud"
    ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.Series]:
        """
        Transform new data using fitted encoders and statistics.
        
        Args:
            df: DataFrame with transactions to transform
            include_target: Whether to return target column
            target_col: Name of target column
        
        Returns:
            Feature DataFrame (and target Series if include_target=True)
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform. Call fit_transform first.")
        
        use_fast = self._should_use_fast_mode(df)
        
        # Ensure data is sorted by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # 1. Transaction-level features (use fitted encoders)
        df, _ = compute_transaction_features(
            df,
            transaction_type_encoder=self.encoders.get("transaction_type_encoder"),
            merchant_category_encoder=self.encoders.get("merchant_category_encoder"),
            fit_encoders=False
        )
        
        # 2. User behavioral aggregates
        if use_fast:
            df = compute_user_aggregates_vectorized(df, self.settings)
        else:
            df = compute_user_aggregates(df, self.settings)
        
        # 3. Deviation features (use fitted merchant stats)
        df, _ = compute_deviation_features(
            df,
            merchant_medians=self.merchant_stats.get("merchant_medians"),
            fit_merchant_stats=False
        )
        
        # 4. Temporal features
        df = compute_temporal_features(df)
        
        # 5. Velocity features
        if use_fast:
            df = compute_velocity_features_vectorized(df, self.settings)
        else:
            df = compute_velocity_features(df, self.settings)
        
        # 6. Cold-start features
        if use_fast:
            df = compute_cold_start_features_vectorized(df, self.settings)
        else:
            df = compute_cold_start_features(df, self.settings)
        
        # 7. Preprocess IEEE features (encode categoricals)
        df = self._preprocess_ieee_features(df, fit=False)
        
        # 8. Extract feature columns and fill missing values
        # Use the feature columns that were fitted during training
        feature_cols = getattr(self, '_fitted_feature_cols', self.settings.feature_columns)
        
        # Ensure all expected columns exist (handle optional/missing features)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan
                
        X = df[feature_cols].copy()
        X = X.fillna(self.fill_values)
        
        if include_target and target_col in df.columns:
            y = df[target_col].astype(int)
            return X, y
        
        return X
    
    def transform_single(
        self,
        transaction: dict,
        user_history: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Transform a single transaction for real-time inference.
        
        This is a lightweight version for API inference where we have
        the user's history pre-loaded.
        
        Args:
            transaction: Dict with transaction fields
            user_history: DataFrame of user's past transactions
        
        Returns:
            Feature DataFrame with single row
        """
        # Create DataFrame with single transaction
        df = pd.DataFrame([transaction])
        
        # If we have user history, prepend it for feature computation
        if user_history is not None and len(user_history) > 0:
            # Combine history with current transaction
            combined = pd.concat([user_history, df], ignore_index=True)
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            
            # Transform all, then extract only the last row (current transaction)
            X = self.transform(combined)
            X = X.iloc[[-1]]  # Last row
        else:
            # No history, transform just the single transaction
            X = self.transform(df)
        
        return X
    
    def save(self, filepath: str | Path) -> None:
        """
        Save the fitted feature engineer to disk.
        
        Args:
            filepath: Path to save the pickle file
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted FeatureEngineer")
        
        state = {
            "encoders": self.encoders,
            "merchant_stats": self.merchant_stats,
            "fill_values": self.fill_values,
            "feature_columns": self.settings.feature_columns,
            "fitted_feature_cols": getattr(self, '_fitted_feature_cols', self.settings.feature_columns),
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, filepath)
        
        print(f"Saved feature engineer to {filepath}")
    
    @classmethod
    def load(cls, filepath: str | Path) -> "FeatureEngineer":
        """
        Load a fitted feature engineer from disk.
        
        Args:
            filepath: Path to the pickle file
        
        Returns:
            Fitted FeatureEngineer instance
        """
        state = joblib.load(filepath)
        
        engineer = cls()
        engineer.encoders = state["encoders"]
        engineer.merchant_stats = state["merchant_stats"]
        engineer.fill_values = state["fill_values"]
        engineer._fitted_feature_cols = state.get("fitted_feature_cols", state.get("feature_columns", []))
        engineer.is_fitted = True
        
        print(f"Loaded feature engineer from {filepath}")
        
        return engineer
    
    def _preprocess_ieee_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Preprocess IEEE-specific features (encode categoricals).
        
        Args:
            df: DataFrame with IEEE features
            fit: Whether to fit new encoders
        
        Returns:
            DataFrame with encoded IEEE features
        """
        from sklearn.preprocessing import LabelEncoder
        
        # List of IEEE categorical features that need encoding
        ieee_categorical_cols = [
            "ieee_card4", "ieee_card5", "ieee_card6",
            "ieee_r_emaildomain", "ieee_devicetype", "ieee_deviceinfo"
        ]
        
        # Only process columns that exist in the dataframe
        cols_to_encode = [col for col in ieee_categorical_cols if col in df.columns]
        
        for col in cols_to_encode:
            encoder_key = f"{col}_encoder"
            
            if fit:
                # Fit new encoder
                encoder = LabelEncoder()
                # Convert to string and handle NaN
                df[col] = df[col].astype(str).replace('nan', 'missing')
                encoder.fit(df[col])
                self.encoders[encoder_key] = encoder
            else:
                # Use existing encoder
                if encoder_key in self.encoders:
                    encoder = self.encoders[encoder_key]
                    df[col] = df[col].astype(str).replace('nan', 'missing')
                    # Handle unseen categories
                    known_classes = set(encoder.classes_)
                    df[col] = df[col].apply(lambda x: x if x in known_classes else 'missing')
            
            # Encode
            if encoder_key in self.encoders:
                df[col] = self.encoders[encoder_key].transform(df[col])
        
        return df
    
    def get_feature_names(self) -> list:
        """Get list of feature column names."""
        return self.settings.feature_columns

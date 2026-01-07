"""
Unit tests for feature engineering pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.pipeline import FeatureEngineer
from src.features.transaction import compute_transaction_features
from src.features.temporal import compute_temporal_features
from src.features.cold_start import compute_cold_start_features


@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    base_time = datetime(2025, 6, 15, 12, 0, 0)
    
    data = [
        # User 1: Normal transactions
        {"transaction_id": "t1", "user_id": "u1", "amount": 50.0, 
         "timestamp": base_time, "transaction_type": "payment",
         "merchant_category": "grocery", "location": "new york", 
         "is_failed": False, "is_fraud": False},
        {"transaction_id": "t2", "user_id": "u1", "amount": 45.0,
         "timestamp": base_time + timedelta(days=1), "transaction_type": "payment",
         "merchant_category": "grocery", "location": "new york",
         "is_failed": False, "is_fraud": False},
        {"transaction_id": "t3", "user_id": "u1", "amount": 500.0,  # Unusual amount
         "timestamp": base_time + timedelta(days=2), "transaction_type": "transfer",
         "merchant_category": "electronics", "location": "los angeles",  # New location
         "is_failed": False, "is_fraud": True},
         
        # User 2: Few transactions (cold start)
        {"transaction_id": "t4", "user_id": "u2", "amount": 100.0,
         "timestamp": base_time + timedelta(days=1), "transaction_type": "purchase",
         "merchant_category": "restaurant", "location": "chicago",
         "is_failed": False, "is_fraud": False},
    ]
    
    return pd.DataFrame(data)


class TestTransactionFeatures:
    """Tests for transaction-level features."""
    
    def test_log_amount(self, sample_transactions):
        """Test log transformation of amount."""
        df, _ = compute_transaction_features(sample_transactions, fit_encoders=True)
        
        # Check log amount is computed
        assert "log_transaction_amount" in df.columns
        
        # Check values
        expected = np.log1p(sample_transactions["amount"])
        np.testing.assert_array_almost_equal(
            df["log_transaction_amount"].values,
            expected.values
        )
    
    def test_is_new_merchant(self, sample_transactions):
        """Test is_new_merchant_for_user feature."""
        df, _ = compute_transaction_features(sample_transactions, fit_encoders=True)
        
        # First transaction for u1 at grocery: new = 1
        assert df.loc[0, "is_new_merchant_for_user"] == 1
        
        # Second transaction for u1 at same grocery: not new = 0
        assert df.loc[1, "is_new_merchant_for_user"] == 0
        
        # Third transaction for u1 at electronics: new = 1
        assert df.loc[2, "is_new_merchant_for_user"] == 1
    
    def test_is_new_location(self, sample_transactions):
        """Test is_new_location_for_user feature."""
        df, _ = compute_transaction_features(sample_transactions, fit_encoders=True)
        
        # First transaction for u1 in new york: new = 1
        assert df.loc[0, "is_new_location_for_user"] == 1
        
        # Second transaction for u1 in new york: not new = 0
        assert df.loc[1, "is_new_location_for_user"] == 0
        
        # Third transaction for u1 in los angeles: new = 1
        assert df.loc[2, "is_new_location_for_user"] == 1


class TestTemporalFeatures:
    """Tests for temporal features."""
    
    def test_hour_of_day(self, sample_transactions):
        """Test hour extraction."""
        df = compute_temporal_features(sample_transactions)
        
        assert "hour_of_day" in df.columns
        assert df.loc[0, "hour_of_day"] == 12  # Noon
    
    def test_is_night_transaction(self, sample_transactions):
        """Test night transaction detection."""
        # Add a night transaction
        df = sample_transactions.copy()
        df.loc[0, "timestamp"] = datetime(2025, 6, 15, 2, 0, 0)  # 2am
        
        df = compute_temporal_features(df)
        
        assert df.loc[0, "is_night_transaction"] == 1  # 2am is night
        assert df.loc[1, "is_night_transaction"] == 0  # 12pm is not night
    
    def test_is_weekend(self, sample_transactions):
        """Test weekend detection."""
        # Sunday
        df = sample_transactions.copy()
        df.loc[0, "timestamp"] = datetime(2025, 6, 15, 12, 0, 0)  # Sunday
        
        df = compute_temporal_features(df)
        
        assert df.loc[0, "is_weekend"] == 1


class TestColdStartFeatures:
    """Tests for cold-start user features."""
    
    def test_transaction_count(self, sample_transactions):
        """Test user transaction count to date."""
        df = compute_cold_start_features(sample_transactions)
        
        # First transaction for u1: count = 0
        assert df.loc[0, "user_transaction_count_to_date"] == 0
        
        # Second transaction for u1: count = 1
        assert df.loc[1, "user_transaction_count_to_date"] == 1
        
        # Third transaction for u1: count = 2
        assert df.loc[2, "user_transaction_count_to_date"] == 2
    
    def test_is_new_user(self, sample_transactions):
        """Test new user flag."""
        df = compute_cold_start_features(sample_transactions)
        
        # All users have < 5 transactions, so all are "new"
        assert df["is_new_user"].all()


class TestFeatureEngineerPipeline:
    """Integration tests for full feature pipeline."""
    
    def test_fit_transform(self, sample_transactions):
        """Test full fit_transform pipeline."""
        engineer = FeatureEngineer()
        X, y = engineer.fit_transform(sample_transactions)
        
        # Check output shape
        assert len(X) == len(sample_transactions)
        assert len(y) == len(sample_transactions)
        
        # Check all expected features are present
        expected_features = engineer.get_feature_names()
        for feat in expected_features:
            assert feat in X.columns, f"Missing feature: {feat}"
        
        # Check no NaN values
        assert not X.isna().any().any(), "Found NaN values in features"
    
    def test_transform_uses_fitted_encoders(self, sample_transactions):
        """Test that transform uses fitted encoders."""
        engineer = FeatureEngineer()
        X_train, _ = engineer.fit_transform(sample_transactions)
        
        # Create new data with same categories
        new_data = sample_transactions.iloc[[0]].copy()
        new_data["transaction_id"] = "new_txn"
        new_data["timestamp"] = datetime(2025, 12, 1)
        
        X_new = engineer.transform(new_data)
        
        # Should work without errors
        assert len(X_new) == 1
    
    def test_save_load(self, sample_transactions, tmp_path):
        """Test save and load functionality."""
        engineer = FeatureEngineer()
        engineer.fit_transform(sample_transactions)
        
        # Save
        save_path = tmp_path / "feature_engineer.pkl"
        engineer.save(save_path)
        
        # Load
        loaded_engineer = FeatureEngineer.load(save_path)
        
        # Should be able to transform
        X = loaded_engineer.transform(sample_transactions.iloc[[0]])
        assert len(X) == 1


class TestNoDataLeakage:
    """Tests to verify no data leakage in feature computation."""
    
    def test_user_aggregates_use_only_past(self, sample_transactions):
        """Verify user aggregates don't use future transactions."""
        from src.features.user_aggregates import compute_user_aggregates
        
        df = sample_transactions.copy()
        df = compute_user_aggregates(df)
        
        # First transaction for u1 should have no mean (or 0/NaN)
        # because there's no history yet
        first_u1 = df[df["user_id"] == "u1"].iloc[0]
        
        # mean_amount_last_7d should be NaN or 0 for first transaction
        assert pd.isna(first_u1["mean_amount_last_7d"]) or first_u1["mean_amount_last_7d"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

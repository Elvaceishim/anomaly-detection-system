"""
Central configuration for the anomaly detection system.

All feature windows, model parameters, and thresholds are defined here
to ensure consistency across training and inference.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureWindows:
    """Time windows for feature computation (in hours)."""
    
    velocity_1h: int = 1
    velocity_24h: int = 24
    short_term_days: int = 7
    medium_term_days: int = 30
    long_term_days: int = 90
    
    @property
    def short_term_hours(self) -> int:
        return self.short_term_days * 24
    
    @property
    def medium_term_hours(self) -> int:
        return self.medium_term_days * 24
    
    @property
    def long_term_hours(self) -> int:
        return self.long_term_days * 24


@dataclass
class ModelConfig:
    """Model training configuration."""
    
    # Logistic Regression
    lr_max_iter: int = 1000
    lr_class_weight: str = "balanced"
    
    # LightGBM
    lgb_n_estimators: int = 200
    lgb_learning_rate: float = 0.05
    lgb_max_depth: int = 6
    lgb_num_leaves: int = 31
    lgb_min_child_samples: int = 20
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    lgb_early_stopping_rounds: int = 50
    
    # Class weight adjustment for LightGBM
    # Multiplier to reduce false positives (lower = fewer FPs)
    lgb_weight_adjustment: float = 0.3
    
    # Random seed for reproducibility
    random_state: int = 42


@dataclass
class ThresholdConfig:
    """Threshold selection configuration."""
    
    # Target precision for threshold selection
    # Higher = fewer false positives, but also fewer true positives
    target_precision: float = 0.80
    
    # Default threshold if target precision cannot be achieved
    default_threshold: float = 0.5


@dataclass
class ColdStartConfig:
    """Configuration for handling new users."""
    
    # Users with fewer than this many transactions are considered "new"
    new_user_threshold: int = 5
    
    # Minimum transactions needed for reliable std calculation
    min_transactions_for_std: int = 2


@dataclass
class Settings:
    """Main settings container."""
    
    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models")
    
    # Sub-configurations
    feature_windows: FeatureWindows = field(default_factory=FeatureWindows)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    threshold_config: ThresholdConfig = field(default_factory=ThresholdConfig)
    cold_start_config: ColdStartConfig = field(default_factory=ColdStartConfig)
    
    # Feature columns (final feature set)
    feature_columns: List[str] = field(default_factory=lambda: [
        # Transaction-level
        "log_transaction_amount",
        "transaction_type_encoded",
        "merchant_category_encoded",
        "is_new_merchant_for_user",
        "is_new_location_for_user",
        
        # User behavioral aggregates
        "mean_amount_last_7d",
        "std_amount_last_30d",
        "transaction_count_last_24h",
        "total_amount_last_7d",
        "unique_merchants_last_30d",
        "unique_locations_last_30d",
        
        # Deviation features
        "amount_zscore_user",
        "amount_vs_user_median_ratio",
        "time_since_last_transaction",
        "amount_percentile_user",
        "amount_vs_merchant_median",
        
        # Temporal features
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_night_transaction",
        
        # Velocity features
        "transactions_last_1h",
        "amount_sum_last_1h",
        "failed_transaction_count_last_24h",
        
        # Cold-start indicators
        "user_transaction_count_to_date",
        "is_new_user",
        
        # IEEE-specific features (card, device, time deltas, counts)
        "ieee_card2", "ieee_card3", "ieee_card4", "ieee_card5", "ieee_card6",
        "ieee_addr2", "ieee_dist1", "ieee_dist2",
        "ieee_r_emaildomain", "ieee_devicetype", "ieee_deviceinfo",
        "ieee_d1", "ieee_d2", "ieee_d3", "ieee_d4", "ieee_d5", "ieee_d10", "ieee_d15",
        "ieee_c1", "ieee_c2", "ieee_c5", "ieee_c6", "ieee_c13", "ieee_c14",
    ])
    
    categorical_columns: List[str] = field(default_factory=lambda: [
        "transaction_type_encoded",
        "merchant_category_encoded",
        "ieee_devicetype",  # Categorical IEEE feature
    ])
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

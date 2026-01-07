"""
Pydantic schemas for transaction data and model outputs.

These schemas ensure type safety and validation across:
- API request/response handling
- Data loading and processing
- Feature engineering pipeline
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class Transaction(BaseModel):
    """
    Raw transaction input schema.
    
    This represents a single transaction as received from the data source
    or API request. All temporal features will be derived from these fields.
    """
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User who made the transaction")
    amount: float = Field(..., ge=0, description="Transaction amount (non-negative)")
    timestamp: datetime = Field(..., description="When the transaction occurred")
    transaction_type: str = Field(..., description="Type: transfer, payment, cash-out, etc.")
    merchant_category: str = Field(..., description="Merchant category code or name")
    location: str = Field(..., description="Transaction location (city or region)")
    is_failed: bool = Field(default=False, description="Whether the transaction failed")
    
    # Optional: fraud label (may be missing for unlabeled data)
    is_fraud: Optional[bool] = Field(default=None, description="Fraud label (if available)")
    
    @field_validator("transaction_type", "merchant_category", "location")
    @classmethod
    def lowercase_strings(cls, v: str) -> str:
        """Normalize string fields to lowercase for consistent encoding."""
        return v.lower().strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_001",
                "user_id": "user_123",
                "amount": 150.00,
                "timestamp": "2025-01-15T14:30:00",
                "transaction_type": "transfer",
                "merchant_category": "electronics",
                "location": "new york",
                "is_failed": False,
                "is_fraud": None
            }
        }


class TransactionFeatures(BaseModel):
    """
    Computed features for a transaction.
    
    All features are computed using only past data (timestamp < current)
    to prevent data leakage.
    """
    
    # Transaction-level features
    log_transaction_amount: float
    transaction_type_encoded: int
    merchant_category_encoded: int
    is_new_merchant_for_user: int  # 0 or 1
    is_new_location_for_user: int  # 0 or 1
    
    # User behavioral aggregates
    mean_amount_last_7d: float
    std_amount_last_30d: float
    transaction_count_last_24h: int
    total_amount_last_7d: float
    unique_merchants_last_30d: int
    unique_locations_last_30d: int
    
    # Deviation features
    amount_zscore_user: float
    amount_vs_user_median_ratio: float
    time_since_last_transaction: float  # in hours
    amount_percentile_user: float  # 0.0 to 1.0
    amount_vs_merchant_median: float
    
    # Temporal features
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int  # 0 or 1
    is_night_transaction: int  # 0 or 1
    
    # Velocity features
    transactions_last_1h: int
    amount_sum_last_1h: float
    failed_transaction_count_last_24h: int
    
    # Cold-start indicators
    user_transaction_count_to_date: int
    is_new_user: int  # 0 or 1


class PredictionOutput(BaseModel):
    """
    Model prediction output schema.
    
    Returned by the inference API for each scored transaction.
    """
    
    transaction_id: str = Field(..., description="Original transaction ID")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Fraud probability (0-1)")
    is_flagged: bool = Field(..., description="Whether transaction exceeds threshold")
    threshold: float = Field(..., description="Decision threshold used")
    
    # Optional: top contributing features for explainability
    top_features: Optional[List[dict]] = Field(
        default=None,
        description="Top features contributing to the score"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_001",
                "risk_score": 0.73,
                "is_flagged": True,
                "threshold": 0.65,
                "top_features": [
                    {"feature": "amount_zscore_user", "value": 3.2, "contribution": 0.15},
                    {"feature": "is_new_merchant_for_user", "value": 1, "contribution": 0.12}
                ]
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""
    
    transactions: List[Transaction] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""
    
    predictions: List[PredictionOutput]
    processed_count: int
    failed_count: int = 0

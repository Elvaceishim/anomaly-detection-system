"""
Pydantic schemas for MCP tool inputs and outputs.

These schemas define the structure of data that flows through
MCP tools, ensuring type safety and clear documentation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime


# ============================================================
# Input Schemas
# ============================================================

class TransactionIdInput(BaseModel):
    """Input for tools that require a transaction ID."""
    transaction_id: str = Field(..., description="The unique transaction identifier")


class UserIdInput(BaseModel):
    """Input for tools that require a user ID."""
    user_id: str = Field(..., description="The unique user identifier")


class HumanDecisionInput(BaseModel):
    """Input for logging a human review decision."""
    transaction_id: str = Field(..., description="The transaction being reviewed")
    decision: Literal["approve", "reject", "escalate"] = Field(
        ..., description="The analyst's decision"
    )
    notes: Optional[str] = Field(
        None, description="Optional notes explaining the decision"
    )
    analyst_id: Optional[str] = Field(
        None, description="ID of the analyst making the decision"
    )


# ============================================================
# Output Schemas
# ============================================================

class TransactionSummary(BaseModel):
    """Summary of a transaction (safe to expose to LLM)."""
    transaction_id: str
    amount: float = Field(..., description="Transaction amount in dollars")
    merchant_category: str = Field(..., description="Category of merchant")
    timestamp: str = Field(..., description="ISO format timestamp")
    location: str = Field(..., description="Coarse location (city level)")
    transaction_type: str
    risk_score: float = Field(..., ge=0, le=1, description="Model's risk score (0-1)")
    is_flagged: bool = Field(..., description="Whether transaction exceeds threshold")
    
    # Explicitly document what is NOT included
    class Config:
        json_schema_extra = {
            "not_included": [
                "full_card_number",
                "user_name", 
                "exact_address",
                "fraud_outcome"
            ]
        }


class UserBehaviorSnapshot(BaseModel):
    """Snapshot of user's typical behavior (aggregated, no raw history)."""
    user_id: str
    transaction_count_30d: int = Field(..., description="Transactions in last 30 days")
    avg_amount_30d: float = Field(..., description="Average transaction amount")
    amount_range: tuple = Field(..., description="(min, max) typical amounts")
    velocity_last_24h: int = Field(..., description="Transactions in last 24 hours")
    unique_merchants_30d: int = Field(..., description="Unique merchants visited")
    unique_locations_30d: int = Field(..., description="Unique locations")
    is_new_user: bool = Field(..., description="Fewer than 5 historical transactions")
    
    class Config:
        json_schema_extra = {
            "not_included": [
                "full_transaction_history",
                "exact_timestamps",
                "past_fraud_flags"
            ]
        }


class AnomalySignals(BaseModel):
    """Anomaly signals for a specific transaction."""
    transaction_id: str
    amount_percentile: float = Field(
        ..., ge=0, le=1, 
        description="Where this amount falls in user's history (0=lowest, 1=highest)"
    )
    amount_zscore: float = Field(
        ..., description="Standard deviations from user's mean"
    )
    velocity_spike: bool = Field(
        ..., description="Unusual number of transactions in short period"
    )
    location_change: bool = Field(
        ..., description="Transaction from new location for this user"
    )
    merchant_change: bool = Field(
        ..., description="Transaction with new merchant category"
    )
    time_anomaly: bool = Field(
        ..., description="Unusual time of day for this user"
    )
    amount_vs_merchant_median: float = Field(
        ..., description="Ratio of amount to typical for this merchant category"
    )


class ModelExplanation(BaseModel):
    """Explanation of why the model assigned this risk score."""
    transaction_id: str
    risk_score: float
    threshold: float
    is_flagged: bool
    top_contributing_features: List[dict] = Field(
        ..., description="Top features that influenced the score"
    )
    explanation_summary: str = Field(
        ..., description="Human-readable summary of the key risk factors"
    )
    
    class Config:
        json_schema_extra = {
            "not_included": [
                "model_weights",
                "exact_thresholds",
                "training_statistics"
            ]
        }


class DecisionLogResult(BaseModel):
    """Confirmation of a logged human decision."""
    success: bool
    transaction_id: str
    decision: str
    logged_at: str
    log_id: str

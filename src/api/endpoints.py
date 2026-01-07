"""
API endpoints for transaction scoring.

Provides:
- Single transaction scoring
- Batch transaction scoring
"""

from fastapi import APIRouter, HTTPException, status
from typing import List
import pandas as pd
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.schemas import (
    Transaction,
    PredictionOutput,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from .app import model_state
from .store import TransactionStore

# Initialize global transaction store
transaction_store = TransactionStore()

router = APIRouter(tags=["predictions"])


def check_model_loaded():
    """Raise error if model is not loaded."""
    if not model_state.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train and save a model first."
        )


@router.post("/predict", response_model=PredictionOutput)
async def predict_single(transaction: Transaction) -> PredictionOutput:
    """
    Score a single transaction for fraud risk.
    
    Returns a risk score between 0 and 1, along with a binary flag
    indicating whether the transaction should be flagged for review.
    
    Example request:
    ```json
    {
        "transaction_id": "txn_001",
        "user_id": "user_123",
        "amount": 500.00,
        "timestamp": "2025-01-15T14:30:00",
        "transaction_type": "transfer",
        "merchant_category": "electronics",
        "location": "new york",
        "is_failed": false
    }
    ```
    """
    check_model_loaded()
    
    try:
        # Convert to dict
        txn_dict = transaction.model_dump()
        
        # Get user history from store
        user_history = transaction_store.get_user_history(transaction.user_id)
        
        # Compute features using history
        X = model_state.feature_engineer.transform_single(
            txn_dict,
            user_history=user_history
        )
        
        # Get prediction
        risk_score = float(model_state.model.predict_proba(X)[0])
        is_flagged = risk_score >= model_state.threshold
        
        # Save transaction to store (after successful prediction)
        transaction_store.add_transaction(txn_dict)
        
        return PredictionOutput(
            transaction_id=transaction.transaction_id,
            risk_score=round(risk_score, 4),
            is_flagged=is_flagged,
            threshold=model_state.threshold
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Score multiple transactions in a single request.
    
    Accepts up to 1000 transactions per batch.
    Returns predictions for all successfully processed transactions.
    """
    check_model_loaded()
    
    predictions = []
    failed_count = 0
    
    try:
        # Convert all transactions to DataFrame
        txn_dicts = [t.model_dump() for t in request.transactions]
        df = pd.DataFrame(txn_dicts)
        
        # Compute features for all
        X = model_state.feature_engineer.transform(df)
        
        # Get predictions
        risk_scores = model_state.model.predict_proba(X)
        
        # Build response
        for i, txn in enumerate(request.transactions):
            try:
                risk_score = float(risk_scores[i])
                predictions.append(PredictionOutput(
                    transaction_id=txn.transaction_id,
                    risk_score=round(risk_score, 4),
                    is_flagged=risk_score >= model_state.threshold,
                    threshold=model_state.threshold
                ))
            except Exception:
                failed_count += 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            processed_count=len(predictions),
            failed_count=failed_count
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns model metadata, threshold, and feature list.
    """
    check_model_loaded()
    
    return {
        "model_type": model_state.metadata.get("model_type"),
        "created_at": model_state.metadata.get("created_at"),
        "threshold": model_state.threshold,
        "metrics": {
            "pr_auc": model_state.metadata.get("pr_auc"),
            "roc_auc": model_state.metadata.get("roc_auc")
        },
        "feature_count": len(model_state.metadata.get("feature_columns", []))
    }


@router.post("/threshold/update")
async def update_threshold(new_threshold: float):
    """
    Update the decision threshold dynamically.
    
    This allows adjusting the sensitivity without retraining.
    
    - Lower threshold = more transactions flagged (higher recall, lower precision)
    - Higher threshold = fewer transactions flagged (lower recall, higher precision)
    
    Args:
        new_threshold: New threshold value (0.0 to 1.0)
    """
    check_model_loaded()
    
    if not 0.0 <= new_threshold <= 1.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Threshold must be between 0.0 and 1.0"
        )
    
    old_threshold = model_state.threshold
    model_state.threshold = new_threshold
    
    return {
        "message": "Threshold updated",
        "old_threshold": old_threshold,
        "new_threshold": new_threshold
    }


@router.post("/store/reset")
async def reset_store():
    """Clear the in-memory transaction store (for testing)."""
    transaction_store.clear()
    return {"message": "Transaction store cleared"}

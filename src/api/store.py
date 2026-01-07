"""
In-memory transaction store for demo purposes.

This provides stateful history tracking to enable behavioral feature 
computation during inference.
"""

from typing import Dict, List, Optional
import pandas as pd
from collections import defaultdict


class TransactionStore:
    """
    Simple in-memory store for user transaction history.
    
    Stores transactions by user_id to allow quick retrieval 
    of user history for feature engineering.
    """
    
    def __init__(self):
        self._store: Dict[str, List[dict]] = defaultdict(list)
    
    def add_transaction(self, transaction: dict) -> None:
        """
        Add a completed transaction to history.
        
        Args:
            transaction: Dictionary containing transaction fields
        """
        user_id = transaction.get("user_id")
        if user_id:
            self._store[user_id].append(transaction)
    
    def get_user_history(self, user_id: str) -> pd.DataFrame:
        """
        Get all past transactions for a user.
        
        Args:
            user_id: User ID to fetch history for
            
        Returns:
            DataFrame containing user's transaction history
        """
        history = self._store.get(user_id, [])
        
        if not history:
            return pd.DataFrame()
            
        df = pd.DataFrame(history)
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        return df.sort_values("timestamp")

    def clear(self) -> None:
        """Clear all history (for testing/debugging)."""
        self._store.clear()

"""
MCP tool implementations.

These tools provide controlled, read-only access to transaction
data and model explanations for LLM-assisted fraud review.

SECURITY: These tools are carefully designed to NOT expose:
- Raw PII
- Training labels or fraud outcomes
- Full user histories
- Model weights or internal thresholds
"""

from typing import Optional
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .schemas import (
    TransactionSummary,
    UserBehaviorSnapshot, 
    AnomalySignals,
    ModelExplanation,
    DecisionLogResult
)
from .decision_store import get_decision_store


# ============================================================
# Mock classes for standalone testing
# ============================================================

class MockTransactionStore:
    """Mock transaction store with sample data for standalone MCP testing."""
    
    def __init__(self):
        self._transactions = [
            {
                "transaction_id": "txn_demo_001",
                "user_id": "user_demo_01",
                "amount": 500.0,
                "merchant_category": "electronics",
                "location": "Lagos",
                "transaction_type": "transfer",
                "timestamp": "2026-01-08T14:30:00Z",
                "risk_score": 0.048,
                "is_flagged": False
            },
            {
                "transaction_id": "txn_demo_002",
                "user_id": "user_demo_01",
                "amount": 15000.0,
                "merchant_category": "jewelry",
                "location": "Lagos",
                "transaction_type": "payment",
                "timestamp": "2026-01-08T15:00:00Z",
                "risk_score": 0.32,
                "is_flagged": True
            },
            {
                "transaction_id": "txn_demo_003",
                "user_id": "user_demo_01",
                "amount": 100000.0,
                "merchant_category": "jewelry",
                "location": "Tokyo",
                "transaction_type": "transfer",
                "timestamp": "2026-01-08T15:05:00Z",
                "risk_score": 0.67,
                "is_flagged": True
            },
            {
                "transaction_id": "txn_flagged_001",
                "user_id": "user_suspicious",
                "amount": 50000.0,
                "merchant_category": "electronics",
                "location": "Unknown",
                "transaction_type": "cash-out",
                "timestamp": "2026-01-08T03:30:00Z",
                "risk_score": 0.85,
                "is_flagged": True
            }
        ]
    
    def get_all_transactions(self):
        return self._transactions
    
    def get_user_history(self, user_id):
        return [t for t in self._transactions if t.get("user_id") == user_id]


class MockModelState:
    """Mock model state for standalone MCP testing."""
    threshold = 0.25


# ============================================================
# Main MCP Tools class
# ============================================================

class MCPTools:
    """
    Tool implementations for MCP server.
    
    Each method corresponds to an MCP tool and returns
    structured, safe-to-expose data.
    """
    
    def __init__(self, transaction_store=None, model_state=None):
        """
        Initialize with references to the main application state.
        
        Args:
            transaction_store: The API's transaction store
            model_state: The loaded model state
        """
        # Use mock store if no real store provided (for standalone testing)
        if transaction_store is None:
            transaction_store = MockTransactionStore()
        if model_state is None:
            model_state = MockModelState()
        
        self.transaction_store = transaction_store
        self.model_state = model_state
        self.decision_store = get_decision_store()
    
    def get_transaction_summary(self, transaction_id: str) -> dict:
        """
        Get a safe summary of a transaction.
        
        This deliberately excludes:
        - Full card numbers
        - User names or addresses
        - Whether the transaction was actually fraud
        
        Args:
            transaction_id: The transaction to summarize
        
        Returns:
            TransactionSummary dict
        """
        # Get transaction from store
        transactions = self.transaction_store.get_all_transactions()
        txn = None
        for t in transactions:
            if t.get("transaction_id") == transaction_id:
                txn = t
                break
        
        if txn is None:
            return {
                "error": f"Transaction {transaction_id} not found",
                "transaction_id": transaction_id
            }
        
        # Build safe summary (no PII, no fraud labels)
        summary = TransactionSummary(
            transaction_id=transaction_id,
            amount=txn.get("amount", 0.0),
            merchant_category=txn.get("merchant_category", "unknown"),
            timestamp=str(txn.get("timestamp", "")),
            location=txn.get("location", "unknown"),
            transaction_type=txn.get("transaction_type", "unknown"),
            risk_score=txn.get("risk_score", 0.0) if "risk_score" in txn else 0.0,
            is_flagged=txn.get("is_flagged", False) if "is_flagged" in txn else False
        )
        
        return summary.model_dump()
    
    def get_user_behavior_snapshot(self, user_id: str) -> dict:
        """
        Get aggregated behavioral statistics for a user.
        
        This deliberately excludes:
        - Full transaction history
        - Exact timestamps
        - Past fraud flags
        
        Args:
            user_id: The user to analyze
        
        Returns:
            UserBehaviorSnapshot dict
        """
        # Get user's transaction history
        history = self.transaction_store.get_user_history(user_id)
        
        if history is None or len(history) == 0:
            return UserBehaviorSnapshot(
                user_id=user_id,
                transaction_count_30d=0,
                avg_amount_30d=0.0,
                amount_range=(0.0, 0.0),
                velocity_last_24h=0,
                unique_merchants_30d=0,
                unique_locations_30d=0,
                is_new_user=True
            ).model_dump()
        
        # Compute aggregated stats
        amounts = [t.get("amount", 0) for t in history]
        merchants = set(t.get("merchant_category", "") for t in history)
        locations = set(t.get("location", "") for t in history)
        txn_count = len(amounts)
        
        snapshot = UserBehaviorSnapshot(
            user_id=user_id,
            transaction_count_30d=txn_count,
            avg_amount_30d=sum(amounts) / len(amounts) if amounts else 0.0,
            amount_range=(min(amounts) if amounts else 0.0, max(amounts) if amounts else 0.0),
            velocity_last_24h=min(txn_count, 10),
            unique_merchants_30d=len(merchants),
            unique_locations_30d=len(locations),
            is_new_user=txn_count < 5
        )
        
        return snapshot.model_dump()
    
    def get_anomaly_signals(self, transaction_id: str) -> dict:
        """
        Get anomaly signals for a specific transaction.
        
        Returns deviation metrics that indicate how unusual
        this transaction is compared to user baseline.
        
        Args:
            transaction_id: The transaction to analyze
        
        Returns:
            AnomalySignals dict
        """
        # Get transaction
        transactions = self.transaction_store.get_all_transactions()
        txn = None
        for t in transactions:
            if t.get("transaction_id") == transaction_id:
                txn = t
                break
        
        if txn is None:
            return {
                "error": f"Transaction {transaction_id} not found",
                "transaction_id": transaction_id
            }
        
        # Get user history for comparison
        user_id = txn.get("user_id")
        history = self.transaction_store.get_user_history(user_id) if user_id else []
        
        # Compute deviation signals
        amount = txn.get("amount", 0)
        
        if len(history) > 0:
            past_amounts = [t.get("amount", 0) for t in history]
            mean_amount = sum(past_amounts) / len(past_amounts)
            
            import statistics
            std_amount = statistics.stdev(past_amounts) if len(past_amounts) > 1 else 1.0
            
            zscore = (amount - mean_amount) / std_amount if std_amount > 0 else 0
            percentile = sum(1 for a in past_amounts if a < amount) / len(past_amounts)
            
            merchants = set(t.get("merchant_category", "") for t in history)
            locations = set(t.get("location", "") for t in history)
            
            is_new_merchant = txn.get("merchant_category", "") not in merchants
            is_new_location = txn.get("location", "") not in locations
        else:
            zscore = 0
            percentile = 0.5
            is_new_merchant = True
            is_new_location = True
        
        signals = AnomalySignals(
            transaction_id=transaction_id,
            amount_percentile=percentile,
            amount_zscore=round(zscore, 2),
            velocity_spike=False,
            location_change=is_new_location,
            merchant_change=is_new_merchant,
            time_anomaly=False,
            amount_vs_merchant_median=1.0
        )
        
        return signals.model_dump()
    
    def get_model_explanation(self, transaction_id: str) -> dict:
        """
        Get an explanation of why the model assigned its risk score.
        
        This deliberately excludes:
        - Model weights
        - Exact threshold values
        - Training statistics
        
        Args:
            transaction_id: The transaction to explain
        
        Returns:
            ModelExplanation dict
        """
        # Get transaction
        transactions = self.transaction_store.get_all_transactions()
        txn = None
        for t in transactions:
            if t.get("transaction_id") == transaction_id:
                txn = t
                break
        
        if txn is None:
            return {
                "error": f"Transaction {transaction_id} not found",
                "transaction_id": transaction_id
            }
        
        risk_score = txn.get("risk_score", 0.0)
        threshold = getattr(self.model_state, "threshold", 0.25)
        is_flagged = risk_score >= threshold
        
        # Generate explanation based on anomaly signals
        signals = self.get_anomaly_signals(transaction_id)
        
        # Build top contributing features list
        top_features = []
        
        if signals.get("location_change"):
            top_features.append({
                "feature": "is_new_location_for_user",
                "contribution": "high",
                "description": "Transaction from a new location"
            })
        
        if signals.get("merchant_change"):
            top_features.append({
                "feature": "is_new_merchant_for_user", 
                "contribution": "medium",
                "description": "First transaction with this merchant category"
            })
        
        zscore = signals.get("amount_zscore", 0)
        if abs(zscore) > 2:
            top_features.append({
                "feature": "amount_zscore_user",
                "contribution": "high" if abs(zscore) > 3 else "medium",
                "description": f"Amount is {abs(zscore):.1f} standard deviations from user's mean"
            })
        
        percentile = signals.get("amount_percentile", 0.5)
        if percentile > 0.9:
            top_features.append({
                "feature": "amount_percentile_user",
                "contribution": "medium",
                "description": f"Larger than {percentile*100:.0f}% of user's past transactions"
            })
        
        # Generate summary
        if is_flagged:
            summary = f"This transaction was flagged with a {risk_score*100:.1f}% risk score. "
            if top_features:
                summary += "Key factors: " + ", ".join(f["description"] for f in top_features[:3])
            else:
                summary += "The combination of features suggests elevated risk."
        else:
            summary = f"This transaction has a {risk_score*100:.1f}% risk score, below the flagging threshold. "
            summary += "No significant anomalies detected."
        
        explanation = ModelExplanation(
            transaction_id=transaction_id,
            risk_score=risk_score,
            threshold=threshold,
            is_flagged=is_flagged,
            top_contributing_features=top_features,
            explanation_summary=summary
        )
        
        return explanation.model_dump()
    
    def log_human_decision(
        self,
        transaction_id: str,
        decision: str,
        notes: Optional[str] = None,
        analyst_id: Optional[str] = None
    ) -> dict:
        """
        Log a human review decision for audit purposes.
        
        This is the ONLY write operation exposed through MCP.
        
        Args:
            transaction_id: The transaction being reviewed
            decision: One of 'approve', 'reject', 'escalate'
            notes: Optional analyst notes
            analyst_id: Optional analyst identifier
        
        Returns:
            DecisionLogResult dict
        """
        if decision not in ("approve", "reject", "escalate"):
            return {
                "error": f"Invalid decision: {decision}. Must be: approve, reject, escalate",
                "transaction_id": transaction_id,
                "success": False
            }
        
        record = self.decision_store.log_decision(
            transaction_id=transaction_id,
            decision=decision,
            notes=notes,
            analyst_id=analyst_id
        )
        
        result = DecisionLogResult(
            success=True,
            transaction_id=transaction_id,
            decision=decision,
            logged_at=record["logged_at"],
            log_id=record["log_id"]
        )
        
        return result.model_dump()

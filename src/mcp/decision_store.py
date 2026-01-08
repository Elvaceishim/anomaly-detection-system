"""
Simple JSON-based storage for human review decisions.

This provides an audit trail of all analyst decisions made
through the MCP interface.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from threading import Lock


class DecisionStore:
    """
    Thread-safe storage for human review decisions.
    
    Decisions are stored in a JSON file for simplicity.
    In production, this would be a database table.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the decision store.
        
        Args:
            storage_path: Path to the JSON storage file.
                         Defaults to data/decisions.json
        """
        if storage_path is None:
            storage_path = Path(__file__).parent.parent.parent / "data" / "decisions.json"
        
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        
        # Initialize file if it doesn't exist
        if not self.storage_path.exists():
            self._write_decisions([])
    
    def _read_decisions(self) -> List[dict]:
        """Read all decisions from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _write_decisions(self, decisions: List[dict]) -> None:
        """Write decisions to storage."""
        with open(self.storage_path, 'w') as f:
            json.dump(decisions, f, indent=2, default=str)
    
    def log_decision(
        self,
        transaction_id: str,
        decision: str,
        notes: Optional[str] = None,
        analyst_id: Optional[str] = None
    ) -> dict:
        """
        Log a human review decision.
        
        Args:
            transaction_id: The transaction being reviewed
            decision: One of 'approve', 'reject', 'escalate'
            notes: Optional notes from the analyst
            analyst_id: Optional identifier for the analyst
        
        Returns:
            The logged decision record
        """
        with self._lock:
            decisions = self._read_decisions()
            
            record = {
                "log_id": str(uuid.uuid4()),
                "transaction_id": transaction_id,
                "decision": decision,
                "notes": notes,
                "analyst_id": analyst_id,
                "logged_at": datetime.utcnow().isoformat() + "Z",
            }
            
            decisions.append(record)
            self._write_decisions(decisions)
            
            return record
    
    def get_decisions_for_transaction(self, transaction_id: str) -> List[dict]:
        """Get all decisions for a specific transaction."""
        decisions = self._read_decisions()
        return [d for d in decisions if d["transaction_id"] == transaction_id]
    
    def get_recent_decisions(self, limit: int = 50) -> List[dict]:
        """Get the most recent decisions."""
        decisions = self._read_decisions()
        return decisions[-limit:]
    
    def clear(self) -> None:
        """Clear all decisions (for testing only)."""
        with self._lock:
            self._write_decisions([])


# Singleton instance
_decision_store: Optional[DecisionStore] = None


def get_decision_store() -> DecisionStore:
    """Get the global decision store instance."""
    global _decision_store
    if _decision_store is None:
        _decision_store = DecisionStore()
    return _decision_store

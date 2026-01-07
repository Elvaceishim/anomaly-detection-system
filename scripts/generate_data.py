#!/usr/bin/env python
"""
Generate synthetic transaction data for development and testing.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --n-transactions 50000 --fraud-rate 0.03
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.synthetic import generate_synthetic_data
from config.settings import get_settings


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data")
    parser.add_argument(
        "--n-users", type=int, default=1000,
        help="Number of unique users (default: 1000)"
    )
    parser.add_argument(
        "--n-transactions", type=int, default=100000,
        help="Total number of transactions (default: 100000)"
    )
    parser.add_argument(
        "--fraud-rate", type=float, default=0.02,
        help="Base fraud rate (default: 0.02)"
    )
    parser.add_argument(
        "--start-date", type=str, default="2025-01-01",
        help="Start date for data (default: 2025-01-01)"
    )
    parser.add_argument(
        "--end-date", type=str, default="2025-12-31",
        help="End date for data (default: 2025-12-31)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: data/transactions.csv)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Get settings for output path
    settings = get_settings()
    output_path = Path(args.output) if args.output else settings.data_dir / "transactions.csv"
    
    print("=" * 60)
    print("GENERATING SYNTHETIC TRANSACTION DATA")
    print("=" * 60)
    print(f"  Users: {args.n_users:,}")
    print(f"  Transactions: {args.n_transactions:,}")
    print(f"  Fraud Rate: {args.fraud_rate:.1%}")
    print(f"  Date Range: {args.start_date} to {args.end_date}")
    print(f"  Output: {output_path}")
    print("=" * 60)
    
    # Generate data
    df = generate_synthetic_data(
        n_users=args.n_users,
        n_transactions=args.n_transactions,
        fraud_rate=args.fraud_rate,
        start_date=args.start_date,
        end_date=args.end_date,
        random_seed=args.seed
    )
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved {len(df):,} transactions to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Print sample
    print("\nSample transactions:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()

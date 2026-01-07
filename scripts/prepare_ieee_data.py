#!/usr/bin/env python
"""
IEEE-CIS Fraud Detection Data Adapter

This script converts the IEEE-CIS Kaggle dataset to the format expected
by our anomaly detection pipeline.

Download instructions:
1. Go to https://www.kaggle.com/c/ieee-fraud-detection/data
2. Accept the competition rules
3. Download train_transaction.csv and train_identity.csv
4. Place them in data/ieee-cis/
5. Run this script: python3 scripts/prepare_ieee_data.py

The IEEE-CIS dataset contains:
- train_transaction.csv: Transaction features (~590K rows, 394 columns)
- train_identity.csv: Identity features (~144K rows, 41 columns)
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings


def load_ieee_data(data_dir: Path) -> pd.DataFrame:
    """
    Load and merge IEEE-CIS transaction and identity data.
    
    Args:
        data_dir: Directory containing the raw IEEE-CIS files
    
    Returns:
        Merged DataFrame
    """
    transaction_file = data_dir / "train_transaction.csv"
    identity_file = data_dir / "train_identity.csv"
    
    if not transaction_file.exists():
        raise FileNotFoundError(
            f"Transaction file not found at {transaction_file}\n"
            "Please download from: https://www.kaggle.com/c/ieee-fraud-detection/data"
        )
    
    print(f"Loading transactions from {transaction_file}...")
    df_trans = pd.read_csv(transaction_file)
    print(f"  Loaded {len(df_trans):,} transactions")
    
    # Identity data is optional (not all transactions have identity info)
    if identity_file.exists():
        print(f"Loading identity data from {identity_file}...")
        df_identity = pd.read_csv(identity_file)
        print(f"  Loaded {len(df_identity):,} identity records")
        
        # Merge on TransactionID
        df = df_trans.merge(df_identity, on="TransactionID", how="left")
        print(f"  Merged dataset: {len(df):,} rows")
    else:
        print("Identity file not found, using transaction data only")
        df = df_trans
    
    return df


def convert_to_pipeline_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert IEEE-CIS format to our pipeline's expected schema.
    
    IEEE-CIS fields -> Our schema:
    - TransactionID -> transaction_id
    - card1 (hashed) -> user_id (proxy)
    - TransactionAmt -> amount
    - TransactionDT -> timestamp (seconds from reference point)
    - ProductCD -> transaction_type
    - P_emaildomain -> merchant_category (proxy)
    - addr1 -> location (proxy)
    - isFraud -> is_fraud
    
    Note: IEEE-CIS data is anonymized, so some mappings are approximate.
    """
    print("Converting to pipeline format...")
    
    # Reference date for timestamp conversion
    # IEEE-CIS TransactionDT is seconds from a reference point
    # The exact date is not disclosed, but late 2017 is commonly assumed
    reference_date = pd.Timestamp("2017-11-30")
    
    # Create mapped DataFrame
    df_mapped = pd.DataFrame({
        # Transaction ID (direct mapping)
        "transaction_id": df["TransactionID"].astype(str),
        
        # User ID - use card1 as user proxy
        # card1 is a hashed card identifier
        "user_id": "user_" + df["card1"].fillna(0).astype(int).astype(str),
        
        # Amount (direct mapping)
        "amount": df["TransactionAmt"],
        
        # Timestamp - convert from seconds offset
        "timestamp": reference_date + pd.to_timedelta(df["TransactionDT"], unit="s"),
        
        # Transaction type - use ProductCD
        # W = digital goods, H = direct goods, C = card present, etc.
        "transaction_type": df["ProductCD"].fillna("unknown").str.lower(),
        
        # Merchant category - use P_emaildomain as proxy
        # This represents the purchaser's email domain, which often
        # correlates with merchant type (gmail = consumer, .edu = education)
        "merchant_category": df["P_emaildomain"].fillna("unknown").str.lower(),
        
        # Location - use addr1 as proxy (billing address region)
        "location": "region_" + df["addr1"].fillna(0).astype(int).astype(str),
        
        # Failed transaction - not directly available, default to False
        "is_failed": False,
        
        # Fraud label (direct mapping)
        "is_fraud": df["isFraud"].astype(bool)
    })
    
    # Sort by timestamp
    df_mapped = df_mapped.sort_values("timestamp").reset_index(drop=True)
    
    return df_mapped


def add_ieee_specific_features(df_original: pd.DataFrame, df_mapped: pd.DataFrame) -> pd.DataFrame:
    """
    Add useful IEEE-CIS specific features that don't need transformation.
    
    These are features from the original dataset that provide additional
    signal beyond what our standard feature engineering provides.
    """
    # Select useful original features to preserve
    useful_cols = [
        # Card info
        "card2", "card3", "card4", "card5", "card6",
        
        # Address info
        "addr2", "dist1", "dist2",
        
        # Email domains
        "R_emaildomain",
        
        # Device info (from identity)
        "DeviceType", "DeviceInfo",
        
        # Time deltas (D features) - these are valuable!
        "D1", "D2", "D3", "D4", "D5", "D10", "D15",
        
        # Count features (C features)
        "C1", "C2", "C5", "C6", "C13", "C14",
    ]
    
    # Only add columns that exist
    available_cols = [c for c in useful_cols if c in df_original.columns]
    
    if available_cols:
        for col in available_cols:
            df_mapped[f"ieee_{col.lower()}"] = df_original[col].values
    
    return df_mapped


def print_dataset_stats(df: pd.DataFrame) -> None:
    """Print statistics about the converted dataset."""
    print("\n" + "=" * 60)
    print("CONVERTED DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal transactions: {len(df):,}")
    print(f"Unique users: {df['user_id'].nunique():,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    fraud_count = df['is_fraud'].sum()
    fraud_rate = df['is_fraud'].mean()
    print(f"\nFraud transactions: {fraud_count:,} ({fraud_rate:.2%})")
    
    print(f"\nTransaction types:")
    print(df['transaction_type'].value_counts().head(10).to_string())
    
    print(f"\nAmount statistics:")
    print(f"  Min: ${df['amount'].min():.2f}")
    print(f"  Max: ${df['amount'].max():,.2f}")
    print(f"  Mean: ${df['amount'].mean():.2f}")
    print(f"  Median: ${df['amount'].median():.2f}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert IEEE-CIS data to pipeline format"
    )
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Directory containing IEEE-CIS CSV files (default: data/ieee-cis/)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: data/transactions_ieee.csv)"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Sample N transactions (for testing with smaller data)"
    )
    parser.add_argument(
        "--include-ieee-features", action="store_true",
        help="Include original IEEE-CIS features in output"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    settings = get_settings()
    input_dir = Path(args.input_dir) if args.input_dir else settings.data_dir / "ieee-cis"
    output_path = Path(args.output) if args.output else settings.data_dir / "transactions_ieee.csv"
    
    print("=" * 60)
    print("IEEE-CIS DATA PREPARATION")
    print("=" * 60)
    print(f"  Input directory: {input_dir}")
    print(f"  Output file: {output_path}")
    if args.sample:
        print(f"  Sampling: {args.sample:,} transactions")
    print("=" * 60)
    
    # Load data
    df_original = load_ieee_data(input_dir)
    
    # Sample if requested
    if args.sample and args.sample < len(df_original):
        print(f"\nSampling {args.sample:,} transactions...")
        # Stratified sample to maintain fraud ratio
        df_fraud = df_original[df_original["isFraud"] == 1]
        df_legit = df_original[df_original["isFraud"] == 0]
        
        fraud_ratio = len(df_fraud) / len(df_original)
        n_fraud = int(args.sample * fraud_ratio)
        n_legit = args.sample - n_fraud
        
        df_original = pd.concat([
            df_fraud.sample(min(n_fraud, len(df_fraud)), random_state=42),
            df_legit.sample(min(n_legit, len(df_legit)), random_state=42)
        ])
    
    # Convert to pipeline format
    df_mapped = convert_to_pipeline_format(df_original)
    
    # Optionally include IEEE-specific features
    if args.include_ieee_features:
        df_mapped = add_ieee_specific_features(df_original, df_mapped)
    
    # Print statistics
    print_dataset_stats(df_mapped)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_mapped.to_csv(output_path, index=False)
    
    print(f"\nSaved {len(df_mapped):,} transactions to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print("\nNext steps:")
    print(f"  1. Train model: python3 scripts/train.py --data {output_path}")
    print("  2. Or run full pipeline with new data")


if __name__ == "__main__":
    main()

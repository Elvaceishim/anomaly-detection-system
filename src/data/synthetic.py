"""
Synthetic data generator for testing and development.

Generates realistic transaction data with:
- Multiple users with varying behavior patterns
- Realistic fraud patterns (large amounts, new merchants, velocity bursts)
- Noisy/incomplete labels (as in real fraud detection)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import random


def generate_synthetic_data(
    n_users: int = 1000,
    n_transactions: int = 100000,
    fraud_rate: float = 0.02,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    label_noise_rate: float = 0.1,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic transaction data for testing.
    
    The generated data includes realistic patterns:
    - Users have baseline spending patterns
    - Fraud transactions tend to be larger amounts
    - Fraud often involves new merchants/locations
    - Fraud may occur in bursts (velocity patterns)
    - Labels are intentionally noisy (some fraud is mislabeled or missing)
    
    Args:
        n_users: Number of unique users
        n_transactions: Total number of transactions to generate
        fraud_rate: Base fraud rate (before label noise)
        start_date: Start of date range
        end_date: End of date range
        label_noise_rate: Fraction of labels that are noisy/missing
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic transactions
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range_days = (end - start).days
    
    # User profiles (each user has a baseline behavior)
    user_profiles = _generate_user_profiles(n_users)
    
    # Merchant categories and locations
    merchant_categories = [
        "grocery", "electronics", "restaurant", "gas_station", "online_retail",
        "clothing", "entertainment", "travel", "utilities", "healthcare"
    ]
    locations = [
        "new york", "los angeles", "chicago", "houston", "phoenix",
        "philadelphia", "san antonio", "san diego", "dallas", "austin"
    ]
    
    # Generate transactions
    transactions = []
    
    for i in range(n_transactions):
        # Select user based on activity distribution (some users more active)
        user_idx = np.random.choice(n_users, p=user_profiles["activity_weight"])
        user_id = f"user_{user_idx:05d}"
        user_profile = {k: v[user_idx] for k, v in user_profiles.items()}
        
        # Generate timestamp
        days_offset = np.random.randint(0, date_range_days)
        hour = np.random.choice(24, p=user_profile["hour_distribution"])
        minute = np.random.randint(0, 60)
        timestamp = start + timedelta(days=days_offset, hours=hour, minutes=minute)
        
        # Determine if this is a fraud transaction
        is_fraud = np.random.random() < fraud_rate
        
        # Generate transaction properties (fraud has different patterns)
        if is_fraud:
            txn = _generate_fraud_transaction(
                user_profile, merchant_categories, locations, timestamp
            )
        else:
            txn = _generate_normal_transaction(
                user_profile, merchant_categories, locations, timestamp
            )
        
        txn["transaction_id"] = f"txn_{i:08d}"
        txn["user_id"] = user_id
        txn["timestamp"] = timestamp
        txn["is_fraud"] = is_fraud
        
        transactions.append(txn)
    
    df = pd.DataFrame(transactions)
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Add label noise (simulate real-world incomplete labels)
    df = _add_label_noise(df, label_noise_rate)
    
    # Add some velocity bursts (multiple transactions in short time)
    df = _add_velocity_bursts(df, burst_rate=0.05)
    
    print(f"Generated {len(df):,} transactions")
    print(f"  Users: {df['user_id'].nunique():,}")
    print(f"  Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"  Failed transactions: {df['is_failed'].mean():.2%}")
    
    return df


def _generate_user_profiles(n_users: int) -> dict:
    """Generate user behavior profiles."""
    
    # Base spending amount (log-normal distribution)
    base_amount_mean = np.random.lognormal(mean=4, sigma=1, size=n_users)
    base_amount_std = base_amount_mean * np.random.uniform(0.2, 0.5, size=n_users)
    
    # Activity level (some users transact more)
    activity_weight = np.random.exponential(scale=1, size=n_users)
    activity_weight = activity_weight / activity_weight.sum()
    
    # Preferred merchants (each user has 2-4 main merchants)
    preferred_merchant_count = np.random.randint(2, 5, size=n_users)
    
    # Preferred locations (each user has 1-3 main locations)
    preferred_location_count = np.random.randint(1, 4, size=n_users)
    
    # Hour distribution (some users are night owls, some morning people)
    hour_distributions = []
    for _ in range(n_users):
        # Random peak hour
        peak_hour = np.random.randint(8, 22)
        hours = np.arange(24)
        probs = np.exp(-0.5 * ((hours - peak_hour) / 4) ** 2)
        probs = probs / probs.sum()
        hour_distributions.append(probs)
    
    return {
        "base_amount_mean": base_amount_mean,
        "base_amount_std": base_amount_std,
        "activity_weight": activity_weight,
        "preferred_merchant_count": preferred_merchant_count,
        "preferred_location_count": preferred_location_count,
        "hour_distribution": hour_distributions,
    }


def _generate_normal_transaction(
    user_profile: dict,
    merchant_categories: list,
    locations: list,
    timestamp: datetime
) -> dict:
    """Generate a normal (non-fraud) transaction."""
    
    # Amount from user's normal distribution
    amount = max(1, np.random.normal(
        user_profile["base_amount_mean"],
        user_profile["base_amount_std"]
    ))
    
    # Usually use preferred merchants
    n_preferred = user_profile["preferred_merchant_count"]
    if np.random.random() < 0.8:  # 80% chance of preferred merchant
        merchant_idx = np.random.randint(0, n_preferred)
    else:
        merchant_idx = np.random.randint(0, len(merchant_categories))
    merchant = merchant_categories[merchant_idx % len(merchant_categories)]
    
    # Usually use preferred locations
    n_preferred_loc = user_profile["preferred_location_count"]
    if np.random.random() < 0.9:  # 90% chance of preferred location
        location_idx = np.random.randint(0, n_preferred_loc)
    else:
        location_idx = np.random.randint(0, len(locations))
    location = locations[location_idx % len(locations)]
    
    # Transaction type
    transaction_type = np.random.choice(
        ["payment", "transfer", "cash_out", "purchase"],
        p=[0.4, 0.2, 0.1, 0.3]
    )
    
    return {
        "amount": round(amount, 2),
        "merchant_category": merchant,
        "location": location,
        "transaction_type": transaction_type,
        "is_failed": np.random.random() < 0.02,  # 2% failure rate
    }


def _generate_fraud_transaction(
    user_profile: dict,
    merchant_categories: list,
    locations: list,
    timestamp: datetime
) -> dict:
    """
    Generate a fraud transaction with suspicious patterns.
    
    Fraud patterns:
    - Larger than normal amounts (1.5x to 5x)
    - Often new merchants
    - Often new locations
    - More cash_out transactions
    - Higher failure rate (failed attempts before success)
    """
    
    # Amount: usually larger than normal
    base_amount = user_profile["base_amount_mean"]
    multiplier = np.random.uniform(1.5, 5.0)
    amount = base_amount * multiplier
    
    # Often new merchant (not in user's preferred list)
    if np.random.random() < 0.7:  # 70% chance of new merchant
        merchant = np.random.choice(merchant_categories)
    else:
        n_preferred = user_profile["preferred_merchant_count"]
        merchant_idx = np.random.randint(0, n_preferred)
        merchant = merchant_categories[merchant_idx % len(merchant_categories)]
    
    # Often new location
    if np.random.random() < 0.6:  # 60% chance of new location
        location = np.random.choice(locations)
    else:
        n_preferred_loc = user_profile["preferred_location_count"]
        location_idx = np.random.randint(0, n_preferred_loc)
        location = locations[location_idx % len(locations)]
    
    # Transaction type (more cash_out in fraud)
    transaction_type = np.random.choice(
        ["payment", "transfer", "cash_out", "purchase"],
        p=[0.2, 0.3, 0.35, 0.15]
    )
    
    return {
        "amount": round(amount, 2),
        "merchant_category": merchant,
        "location": location,
        "transaction_type": transaction_type,
        "is_failed": np.random.random() < 0.15,  # Higher failure rate
    }


def _add_label_noise(df: pd.DataFrame, noise_rate: float) -> pd.DataFrame:
    """
    Add label noise to simulate real-world conditions.
    
    In practice:
    - Some fraud is never detected (false negatives in labels)
    - Some legitimate transactions are incorrectly labeled as fraud
    - Some labels are simply missing
    """
    df = df.copy()
    
    n_noisy = int(len(df) * noise_rate)
    noisy_indices = np.random.choice(df.index, size=n_noisy, replace=False)
    
    # For simplicity, flip some labels
    for idx in noisy_indices:
        if np.random.random() < 0.7:
            # Flip the label
            df.loc[idx, "is_fraud"] = not df.loc[idx, "is_fraud"]
        else:
            # Could set to None for missing, but keep as bool for simplicity
            pass
    
    return df


def _add_velocity_bursts(df: pd.DataFrame, burst_rate: float) -> pd.DataFrame:
    """
    Add velocity bursts where users have multiple transactions in short time.
    
    Some of these bursts are fraud (attacker trying to drain account).
    """
    df = df.copy()
    
    # Group by user
    for user_id in df["user_id"].unique():
        if np.random.random() > burst_rate:
            continue
        
        # Add a burst of 3-5 transactions within an hour
        user_mask = df["user_id"] == user_id
        if user_mask.sum() == 0:
            continue
        
        # Pick a random transaction to anchor the burst
        user_indices = df[user_mask].index.tolist()
        anchor_idx = np.random.choice(user_indices)
        anchor_time = df.loc[anchor_idx, "timestamp"]
        
        # Create burst transactions close in time
        burst_size = np.random.randint(3, 6)
        for i in range(burst_size):
            new_time = anchor_time + timedelta(minutes=np.random.randint(1, 60))
            
            # Find transactions close to this time and adjust
            time_diff = (df["timestamp"] - new_time).abs()
            close_mask = (time_diff < timedelta(hours=2)) & user_mask
            
            if close_mask.sum() > 0:
                close_idx = df[close_mask].index[0]
                df.loc[close_idx, "timestamp"] = new_time
    
    # Re-sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    # Generate and save sample data
    df = generate_synthetic_data()
    
    output_path = "data/transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

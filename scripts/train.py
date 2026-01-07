#!/usr/bin/env python
"""
End-to-end training pipeline for fraud detection model.

This script:
1. Loads transaction data
2. Performs temporal train/val/test split
3. Engineers features
4. Trains baseline models (Logistic Regression + LightGBM)
5. Selects optimal threshold
6. Evaluates on test set
7. Saves model artifacts

Usage:
    python scripts/train.py
    python scripts/train.py --data data/transactions.csv --target-precision 0.85
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from src.data.loader import load_transactions, temporal_split
from src.features.pipeline import FeatureEngineer
from src.models.baseline import LogisticRegressionBaseline, LightGBMBaseline
from src.models.evaluation import evaluate_model
from src.models.threshold import select_threshold, get_threshold_analysis
from src.models.serialization import save_model_artifacts


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to transaction data (default: data/transactions.csv)"
    )
    parser.add_argument(
        "--train-end", type=str, default="2025-09-01",
        help="End date for training set (default: 2025-09-01)"
    )
    parser.add_argument(
        "--val-end", type=str, default="2025-11-01",
        help="End date for validation set (default: 2025-11-01)"
    )
    parser.add_argument(
        "--target-precision", type=float, default=None,
        help="Target precision for threshold selection (default: from settings)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save model artifacts (default: models/)"
    )
    parser.add_argument(
        "--skip-lr", action="store_true",
        help="Skip Logistic Regression baseline"
    )
    
    args = parser.parse_args()
    
    # Load settings
    settings = get_settings()
    
    # Resolve paths
    data_path = Path(args.data) if args.data else settings.data_dir / "transactions.csv"
    output_dir = Path(args.output_dir) if args.output_dir else settings.models_dir
    target_precision = args.target_precision or settings.threshold_config.target_precision
    
    print("=" * 70)
    print("TRANSACTION ANOMALY DETECTION - MODEL TRAINING")
    print("=" * 70)
    print(f"  Data: {data_path}")
    print(f"  Train end: {args.train_end}")
    print(f"  Val end: {args.val_end}")
    print(f"  Target precision: {target_precision:.0%}")
    print(f"  Output: {output_dir}")
    print("=" * 70)
    
    # ================================================================
    # 1. LOAD DATA
    # ================================================================
    print("\n[1/6] Loading data...")
    
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        print("Run 'python scripts/generate_data.py' to create synthetic data.")
        sys.exit(1)
    
    df = load_transactions(data_path)
    print(f"Loaded {len(df):,} transactions")
    
    # ================================================================
    # 2. TEMPORAL SPLIT
    # ================================================================
    print("\n[2/6] Performing temporal split...")
    
    train_df, val_df, test_df = temporal_split(
        df,
        train_end=args.train_end,
        val_end=args.val_end
    )
    
    # ================================================================
    # 3. FEATURE ENGINEERING
    # ================================================================
    print("\n[3/6] Engineering features...")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(settings)
    
    # Fit on training data
    X_train, y_train = feature_engineer.fit_transform(train_df)
    
    # Transform validation and test data
    X_val, y_val = feature_engineer.transform(val_df, include_target=True)
    X_test, y_test = feature_engineer.transform(test_df, include_target=True)
    
    print(f"  Train features shape: {X_train.shape}")
    print(f"  Val features shape: {X_val.shape}")
    print(f"  Test features shape: {X_test.shape}")
    
    # ================================================================
    # 4. TRAIN BASELINE MODELS
    # ================================================================
    print("\n[4/6] Training baseline models...")
    
    # Train Logistic Regression (if not skipped)
    lr_model = None
    if not args.skip_lr:
        print("\n--- Logistic Regression ---")
        lr_model = LogisticRegressionBaseline(settings)
        lr_model.fit(X_train, y_train, X_val, y_val)
        
        # Feature importance
        print("\nTop 10 Features (Logistic Regression):")
        lr_importance = lr_model.get_feature_importance(X_train.columns.tolist())
        print(lr_importance.head(10).to_string(index=False))
    
    # Train LightGBM
    print("\n--- LightGBM ---")
    lgb_model = LightGBMBaseline(settings)
    lgb_model.fit(
        X_train, y_train, X_val, y_val,
        categorical_features=settings.categorical_columns
    )
    
    # Feature importance
    print("\nTop 10 Features (LightGBM):")
    lgb_importance = lgb_model.get_feature_importance(X_train.columns.tolist())
    print(lgb_importance.head(10).to_string(index=False))
    
    # ================================================================
    # 5. THRESHOLD SELECTION
    # ================================================================
    print("\n[5/6] Selecting threshold...")
    
    # Use LightGBM for threshold selection (typically better)
    val_proba = lgb_model.predict_proba(X_val)
    
    threshold, threshold_metrics = select_threshold(
        y_val, val_proba,
        target_precision=target_precision,
        settings=settings
    )
    
    # Show threshold analysis
    print("\nThreshold Analysis (Validation Set):")
    analysis = get_threshold_analysis(y_val, val_proba)
    print(analysis[["threshold", "precision", "recall", "false_positive_rate"]].to_string(index=False))
    
    # ================================================================
    # 6. FINAL EVALUATION
    # ================================================================
    print("\n[6/6] Final evaluation on test set...")
    
    test_proba = lgb_model.predict_proba(X_test)
    test_metrics = evaluate_model(
        y_test, test_proba,
        threshold=threshold,
        split_name="Test"
    )
    
    # Compare with Logistic Regression if trained
    if lr_model is not None:
        print("\n--- Logistic Regression on Test Set ---")
        lr_test_proba = lr_model.predict_proba(X_test)
        lr_metrics = evaluate_model(
            y_test, lr_test_proba,
            threshold=threshold,
            split_name="Test (LR)"
        )
    
    # ================================================================
    # 7. SAVE ARTIFACTS
    # ================================================================
    print("\n[7/7] Saving model artifacts...")
    
    save_model_artifacts(
        model=lgb_model,
        feature_engineer=feature_engineer,
        threshold=threshold,
        metrics=test_metrics,
        output_dir=output_dir,
        model_name="fraud_detection"
    )
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model: LightGBM")
    print(f"  Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Precision: {test_metrics['precision']:.2%}")
    print(f"  Recall: {test_metrics['recall']:.2%}")
    print(f"  Artifacts saved to: {output_dir}")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Start API: uvicorn src.api.app:app --reload")
    print("  2. Test: curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{...}'")
    print("  3. View docs: http://localhost:8000/docs")


if __name__ == "__main__":
    main()

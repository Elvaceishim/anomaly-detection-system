"""
Baseline models for transaction anomaly detection.

This module provides two baseline models:
1. LogisticRegressionBaseline - Interpretable, fast, good for debugging
2. LightGBMBaseline - More powerful, handles feature interactions

Both models are designed to handle class imbalance and output calibrated probabilities.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import get_settings, Settings


class LogisticRegressionBaseline:
    """
    Logistic Regression baseline model.
    
    Why use this:
    - Interpretable coefficients (explain why transactions were flagged)
    - Fast training and inference
    - Naturally calibrated probabilities
    - Good debugging baseline (if this fails, features are wrong)
    
    Features are scaled before training (required for logistic regression).
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the model.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted = False
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "LogisticRegressionBaseline":
        """
        Fit the model on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for logging only)
            y_val: Validation labels (for logging only)
        
        Returns:
            self
        """
        print("Training Logistic Regression baseline...")
        
        config = self.settings.model_config
        
        # Scale features (required for logistic regression)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize model with class weighting
        self.model = LogisticRegression(
            class_weight=config.lr_class_weight,
            max_iter=config.lr_max_iter,
            solver="lbfgs",
            random_state=config.random_state,
            n_jobs=-1
        )
        
        # Fit
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Log training performance
        train_proba = self.predict_proba(X_train)
        from .evaluation import compute_metrics
        train_metrics = compute_metrics(y_train, train_proba)
        print(f"  Train PR-AUC: {train_metrics['pr_auc']:.4f}")
        print(f"  Train ROC-AUC: {train_metrics['roc_auc']:.4f}")
        
        # Log validation performance if provided
        if X_val is not None and y_val is not None:
            val_proba = self.predict_proba(X_val)
            val_metrics = compute_metrics(y_val, val_proba)
            print(f"  Val PR-AUC: {val_metrics['pr_auc']:.4f}")
            print(f"  Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability for each transaction.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of probabilities (fraud class)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary fraud labels.
        
        Args:
            X: Feature DataFrame
            threshold: Decision threshold
        
        Returns:
            Array of binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance (coefficients).
        
        Args:
            feature_names: List of feature names
        
        Returns:
            DataFrame with features sorted by absolute coefficient
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = pd.DataFrame({
            "feature": feature_names,
            "coefficient": self.model.coef_[0]
        })
        
        return importance.sort_values("coefficient", key=abs, ascending=False)


class LightGBMBaseline:
    """
    LightGBM baseline model.
    
    Why use this:
    - Handles mixed feature types (categorical + continuous)
    - Captures non-linear relationships and feature interactions
    - Robust to scale (no normalization needed)
    - Native handling of missing values
    - Fast training with early stopping
    
    Uses scale_pos_weight to handle class imbalance, with adjustment
    factor to control false positive rate.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the model.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        self.model: Optional[lgb.LGBMClassifier] = None
        self.is_fitted = False
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        categorical_features: Optional[list] = None
    ) -> "LightGBMBaseline":
        """
        Fit the model on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels (for early stopping)
            categorical_features: List of categorical feature names
        
        Returns:
            self
        """
        print("Training LightGBM baseline...")
        
        config = self.settings.model_config
        
        # Calculate class weight adjustment
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        base_weight = n_neg / n_pos
        
        # Adjust weight to reduce false positives
        # Lower weight = fewer false positives, but also fewer true positives
        adjusted_weight = base_weight * config.lgb_weight_adjustment
        
        print(f"  Class ratio (neg:pos): {base_weight:.1f}:1")
        print(f"  Adjusted weight: {adjusted_weight:.1f}:1")
        
        # Initialize model
        self.model = lgb.LGBMClassifier(
            n_estimators=config.lgb_n_estimators,
            learning_rate=config.lgb_learning_rate,
            max_depth=config.lgb_max_depth,
            num_leaves=config.lgb_num_leaves,
            min_child_samples=config.lgb_min_child_samples,
            subsample=config.lgb_subsample,
            colsample_bytree=config.lgb_colsample_bytree,
            scale_pos_weight=adjusted_weight,
            random_state=config.random_state,
            verbose=-1,
            n_jobs=-1
        )
        
        # Prepare categorical feature indices
        cat_indices = None
        if categorical_features:
            cat_indices = [
                list(X_train.columns).index(col)
                for col in categorical_features
                if col in X_train.columns
            ]
        
        # Fit with early stopping if validation data provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="average_precision",
                callbacks=[
                    lgb.early_stopping(config.lgb_early_stopping_rounds, verbose=False)
                ]
            )
            print(f"  Best iteration: {self.model.best_iteration_}")
        else:
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        
        # Log training performance
        train_proba = self.predict_proba(X_train)
        from .evaluation import compute_metrics
        train_metrics = compute_metrics(y_train, train_proba)
        print(f"  Train PR-AUC: {train_metrics['pr_auc']:.4f}")
        print(f"  Train ROC-AUC: {train_metrics['roc_auc']:.4f}")
        
        # Log validation performance if provided
        if X_val is not None and y_val is not None:
            val_proba = self.predict_proba(X_val)
            val_metrics = compute_metrics(y_val, val_proba)
            print(f"  Val PR-AUC: {val_metrics['pr_auc']:.4f}")
            print(f"  Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability for each transaction.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of probabilities (fraud class)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary fraud labels.
        
        Args:
            X: Feature DataFrame
            threshold: Decision threshold
        
        Returns:
            Array of binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance (gain-based).
        
        Args:
            feature_names: List of feature names
        
        Returns:
            DataFrame with features sorted by importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": self.model.feature_importances_
        })
        
        return importance.sort_values("importance", ascending=False)

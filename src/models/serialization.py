"""
Model serialization utilities.

Saves and loads all artifacts needed for inference:
- Trained model
- Feature engineer (encoders, statistics)
- Threshold
- Metadata
"""

import joblib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import get_settings


def save_model_artifacts(
    model: Any,
    feature_engineer: Any,
    threshold: float,
    metrics: Dict[str, float],
    output_dir: str | Path,
    model_name: str = "fraud_detection"
) -> Path:
    """
    Save all artifacts needed for model inference.
    
    Saves:
    - model.pkl: Trained model
    - feature_engineer.pkl: Feature encoder and statistics
    - threshold.json: Decision threshold and metrics
    - metadata.json: Training metadata
    
    Args:
        model: Trained model object
        feature_engineer: Fitted FeatureEngineer
        threshold: Selected decision threshold
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save artifacts
        model_name: Name prefix for artifacts
    
    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"{model_name}_model.pkl"
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    
    # Save feature engineer
    fe_path = output_dir / f"{model_name}_feature_engineer.pkl"
    feature_engineer.save(fe_path)
    
    # Save threshold and metrics
    threshold_data = {
        "threshold": threshold,
        "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
    }
    threshold_path = output_dir / f"{model_name}_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"Saved threshold to {threshold_path}")
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "created_at": datetime.now().isoformat(),
        "threshold": threshold,
        "feature_columns": get_settings().feature_columns,
        "pr_auc": metrics.get("pr_auc"),
        "roc_auc": metrics.get("roc_auc")
    }
    metadata_path = output_dir / f"{model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    return output_dir


def load_model_artifacts(
    model_dir: str | Path,
    model_name: str = "fraud_detection"
) -> Dict[str, Any]:
    """
    Load all artifacts needed for model inference.
    
    Args:
        model_dir: Directory containing saved artifacts
        model_name: Name prefix for artifacts
    
    Returns:
        Dictionary with model, feature_engineer, threshold, and metadata
    """
    model_dir = Path(model_dir)
    
    # Load model
    model_path = model_dir / f"{model_name}_model.pkl"
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Load feature engineer
    fe_path = model_dir / f"{model_name}_feature_engineer.pkl"
    from src.features.pipeline import FeatureEngineer
    feature_engineer = FeatureEngineer.load(fe_path)
    
    # Load threshold
    threshold_path = model_dir / f"{model_name}_threshold.json"
    with open(threshold_path, "r") as f:
        threshold_data = json.load(f)
    threshold = threshold_data["threshold"]
    print(f"Loaded threshold: {threshold}")
    
    # Load metadata
    metadata_path = model_dir / f"{model_name}_metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    return {
        "model": model,
        "feature_engineer": feature_engineer,
        "threshold": threshold,
        "metrics": threshold_data.get("metrics", {}),
        "metadata": metadata
    }

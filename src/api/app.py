"""
FastAPI application for transaction anomaly detection.

This module creates the FastAPI app with:
- CORS middleware for cross-origin requests
- Health check endpoint
- Lifespan management for model loading
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Global state for loaded model artifacts
class ModelState:
    model = None
    feature_engineer = None
    threshold: float = 0.5
    metadata: dict = {}
    is_loaded: bool = False


model_state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for loading model on startup.
    
    Loads model artifacts when the server starts and cleans up on shutdown.
    """
    # Startup: Load model artifacts
    model_dir = Path(__file__).parent.parent.parent / "models"
    
    if (model_dir / "fraud_detection_model.pkl").exists():
        print(f"Loading model artifacts from {model_dir}...")
        try:
            from src.models.serialization import load_model_artifacts
            artifacts = load_model_artifacts(model_dir)
            
            model_state.model = artifacts["model"]
            model_state.feature_engineer = artifacts["feature_engineer"]
            model_state.threshold = artifacts["threshold"]
            model_state.metadata = artifacts["metadata"]
            model_state.is_loaded = True
            
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("API will return errors until model is trained and saved.")
    else:
        print(f"No model found at {model_dir}")
        print("Run 'python scripts/train.py' to train a model first.")
    
    yield  # Server is running
    
    # Shutdown: Cleanup
    model_state.model = None
    model_state.feature_engineer = None
    model_state.is_loaded = False
    print("Model unloaded.")


# Create FastAPI app
app = FastAPI(
    title="Transaction Anomaly Detection API",
    description="""
    API for scoring financial transactions for fraud risk.
    
    ## Overview
    
    This API provides risk scores for transactions based on:
    - Transaction amount and type
    - User behavioral patterns
    - Temporal patterns
    - Velocity/burst detection
    
    ## Endpoints
    
    - `POST /predict` - Score a single transaction
    - `POST /predict/batch` - Score multiple transactions
    - `GET /health` - Health check
    
    ## Response
    
    Each transaction receives:
    - `risk_score`: 0.0 to 1.0 probability of fraud
    - `is_flagged`: Whether the score exceeds the threshold
    - `threshold`: The decision threshold used
    """,
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
from .endpoints import router
app.include_router(router)


@app.get("/")
async def root():
    """Serve the dashboard."""
    static_dir = Path(__file__).parent.parent / "static"
    return FileResponse(static_dir / "index.html")


# Mount static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API and loaded model.
    """
    return {
        "status": "healthy",
        "model_loaded": model_state.is_loaded,
        "threshold": model_state.threshold if model_state.is_loaded else None,
        "model_type": model_state.metadata.get("model_type") if model_state.is_loaded else None
    }

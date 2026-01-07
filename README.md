# Transaction Anomaly Detection System

A machine learning system that flags suspicious financial transactions for human review.

## Overview

This system provides:
- **Feature engineering pipeline** with temporal safety (no data leakage)
- **Baseline ML models** (Logistic Regression + LightGBM)
- **FastAPI inference API** for batch and single-transaction scoring
- **Threshold-based decision logic** optimized for precision

## Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Primary metric | PR-AUC | Better than ROC-AUC for imbalanced data |
| Imbalance handling | Class weights + threshold tuning | Simple, effective, no synthetic data |
| Train/test split | Temporal (time-based) | Prevents leakage, simulates production |
| False positive priority | High precision threshold | FPs are more costly than FNs |

## Project Structure

```
anomaly-detection-system/
├── config/settings.py           # Configuration
├── src/
│   ├── data/                    # Data schemas and loading
│   ├── features/                # Feature engineering
│   ├── models/                  # ML models and evaluation
│   └── api/                     # FastAPI inference
├── scripts/
│   ├── generate_data.py         # Generate synthetic data
│   └── train.py                 # Training pipeline
└── tests/                       # Unit tests
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data (for testing)

```bash
python scripts/generate_data.py
```

### 3. Train Models

```bash
python scripts/train.py
```

### 4. Start API Server

```bash
uvicorn src.api.app:app --reload --port 8000
```

### 5. Make Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_001",
    "user_id": "user_123",
    "amount": 500.00,
    "timestamp": "2025-01-15T14:30:00",
    "transaction_type": "transfer",
    "merchant_category": "electronics",
    "location": "New York",
    "is_failed": false
  }'
```

## Feature Set

### Transaction-Level
- `log_transaction_amount` - Log-scaled amount
- `transaction_type_encoded` - Encoded transaction type
- `merchant_category_encoded` - Encoded merchant category
- `is_new_merchant_for_user` - First time with this merchant
- `is_new_location_for_user` - First time in this location

### User Behavioral Aggregates
- `mean_amount_last_7d` - Average amount (7 days)
- `std_amount_last_30d` - Amount variability (30 days)
- `transaction_count_last_24h` - Transaction count (24 hours)
- `total_amount_last_7d` - Sum of amounts (7 days)
- `unique_merchants_last_30d` - Distinct merchants (30 days)
- `unique_locations_last_30d` - Distinct locations (30 days)

### Deviation Features
- `amount_zscore_user` - Z-score vs user history
- `amount_vs_user_median_ratio` - Ratio to user median
- `time_since_last_transaction` - Gap since previous
- `amount_percentile_user` - Percentile in user history
- `amount_vs_merchant_median` - Ratio to merchant median

### Temporal Features
- `hour_of_day`, `day_of_week`, `is_weekend`, `is_night_transaction`

### Velocity Features
- `transactions_last_1h` - Count in past hour
- `amount_sum_last_1h` - Sum in past hour
- `failed_transaction_count_last_24h` - Failed attempts

### Cold-Start Indicators
- `user_transaction_count_to_date` - Total user history
- `is_new_user` - Flag for users with < 5 transactions

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Score single transaction |
| `/predict/batch` | POST | Score multiple transactions |
| `/health` | GET | Health check |
| `/store/reset` | POST | Clear transaction history (dev only) |

## Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

### Manual Docker Build

```bash
# Build image
docker build -t anomaly-detection-api .

# Run container
docker run -d -p 8000:8000 --name fraud-api anomaly-detection-api

# Test
curl http://localhost:8000/health
```

### Production Considerations

- **Model Updates**: Models are mounted as a volume (`./models:/app/models:ro`), so you can update models without rebuilding the image.
- **Scaling**: Use `docker-compose up --scale api=3` with a load balancer for horizontal scaling.
- **Persistence**: The in-memory transaction store resets on container restart. For production, integrate Redis or a database.

### Deploy to Render (Recommended)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) and sign up
3. Click **New > Web Service**
4. Connect your GitHub repo
5. Render will auto-detect the Dockerfile
6. Click **Create Web Service**

Your API will be live at `https://your-service.onrender.com`

> **Note**: Free tier services spin down after 15 min of inactivity. First request after sleep takes ~30 seconds.

## License

MIT

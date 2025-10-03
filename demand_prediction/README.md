# 🚖 Taxi Demand Forecasting System

A production-ready end-to-end solution for forecasting taxi demand using LightGBM, featuring advanced data filtering from ClickHouse, iterative multi-step forecasting, and an interactive Streamlit dashboard.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Model Development](#-model-development)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Demo](#-demo-video)

---

## Overview

This project delivers a complete forecasting pipeline consisting of:

1. **Model Development (`model_dev/`)** – Data loading from ClickHouse, feature engineering, LightGBM training, and iterative forecasting
2. **Streamlit Dashboard (`streamlit_app/`)** – Interactive UI for exploring historical data and generating forecasts

### What Makes This System Unique

- **Iterative Multi-Step Forecasting**: Uses previous predictions as features for subsequent forecasts
- **Group-Based Modeling**: Encode each Zone × Borough × Service Type combination to allow the model to differentiate trip counts across groups.
- **Advanced Data Filtering**: Use specific date ranges and minimum data thresholds, excluding groups with insufficient data to avoid unreliable forecasts.
- **60% MAE Improvement** over baseline models
- **Real-Time Predictions** suitable for production deployment

---

## Key Features

### 🎯 Intelligent Forecasting
- Predicts hourly taxi demand by **pickup zone**, **borough**, and **service type**
- **Iterative forecasting**: Each prediction uses previous predictions as lag features
- Captures trends, seasonality, and demand patterns unique to each group
- Handles missing hours through intelligent interpolation

### 🔧 Advanced Data Management
- **ClickHouse Integration**: Direct connection to data warehouse
- **Date Range Filtering**: Train models on specific time periods (e.g., 2022-01-01 to 2025-08-31)
- **Group Volume Filtering**: Only train on groups with sufficient data (configurable minimum hours threshold)
- **Automatic Interpolation**: Fills missing hourly records for complete time series

### ⚡ High Performance
- **LightGBM** with categorical feature support and early stopping
- **Parallel Processing**: Multi-core CPU utilization
- **Memory Efficient**: Chunked data loading and garbage collection
- **Fast Inference**: <100ms prediction latency per group

### 📊 Rich Feature Engineering
- **Temporal Features**: Hour, day of week, weekend indicators, rush hour flags
- **Lag Features**: 24-hour and 168-hour (weekly) lags
- **Rolling Statistics**: Multiple window sizes (24h, 168h)
- **Cyclical Encoding**: Sin/cos transformations for periodic features
- **Group Statistics**: Mean trips per group (density clustering)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  ClickHouse Data Warehouse                              │
│  ↓ Date filtering (start_date → end_date)               │
│  ↓ Optional: Filter by specific groups                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Data Processing (data_loader.py)                       │
│  • Calculate hours per group                            │
│  • Filter groups (min_hours threshold)                  │
│  • Combine group columns into group_id                  │
│  • Interpolate missing hours                            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Feature Engineering (feature_engineer.py)              │
│  • Time features (hour, day, cyclical encoding)         │
│  • Lag features (24h, 168h)                             │
│  • Rolling statistics (multiple windows)                │
│  • Group statistics (density clustering)                │
│  • Train/test split by time (last 8 months = test)      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Categorical Encoding (encoder.py)                      │
│  • Native LightGBM categorical support                  │
│  • time_of_day, group_id as categories                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  LightGBM Training (pipeline.py)                        │
│  • Train on train split with validation                 │
│  • Early stopping (50 rounds)                           │
│  • Evaluate performance                                 │
│  • Retrain on full data with best iteration             │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Iterative Forecasting (forecast.py)                    │
│  • Load trained model                                   │
│  • For each time step:                                  │
│    - Apply feature engineering                          │
│    - Generate prediction                                │
│    - Use prediction for next step's lag features        │
│  • Generate confidence intervals (±20%)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Model Development

### Project Structure

```
demand_prediction/model_dev/
├── logger.py                # Logging utilities with UTF-8 support
├── config.py                # Configuration (data, model, paths, ClickHouse)
├── data_loader.py           # ClickHouse data loading with filtering
├── feature_engineer.py      # Feature extraction and transformation
├── encoder.py               # Categorical encoding for LightGBM
├── pipeline.py              # Training orchestration
├── run_pipeline.py          # Training entry point
├── forecast.py              # Iterative forecasting engine
├── test_pipeline.py         # Pipeline testing utilities
└── test_forecaster.py       # Forecaster testing utilities

models/
├── encoders/                # Saved encoders
│   └── category_mappings.pkl
├── LIGHTGBM_model.pkl       # Trained model
└── pipeline_artifacts.pkl   # Feature names and metadata

results/
├── model_results.csv        # Training performance metrics
└── feature_importances/
    └── feature_importances.csv
```

### Training the Model

#### Basic Training

```bash
python demand_prediction/model_dev/run_pipeline.py
```

This will use default configuration from `config.py`:
- Date range: 2022-01-01 to 2025-08-31
- Minimum hours per group: 24,102 hours
- Test period: Last 8 months

#### Configuration Options

Edit `model_dev/config.py`:

```python
# ClickHouse Connection
CLICKHOUSE_CONFIG = {
    "host": "localhost",
    "port": 8123,
    "username": "your_username",
    "password": "your_password",
    "database": "data_warehouse"
}

# Table and Date Range
TABLE_CONFIG = {
    "table": "mart_demand_prediction",
    "start_date": "2022-01-01",
    "end_date": "2025-08-31"
}

# Data Configuration
DATA_CONFIG = {
    "timestamp_col": "pickup_datetime",
    "target_col": "total_trips",
    "group_cols": ["pickup_zone", "pickup_borough", "service_type"],
    "group_col": "group_id",
    
    # Feature engineering
    "lag_features": [24, 168],  # 1 day, 1 week
    "rolling_windows": [24, 168],
    
    # Train/test split
    "test_months": 8,  # Last 8 months for testing
    "frequency": "h"   # Hourly frequency
}

# LightGBM Parameters
MODEL_CONFIG = {
    "models": {
        "LIGHTGBM": {
            "n_estimators": 5000,
            "learning_rate": 0.05,
            "num_leaves": 500,
            "min_child_samples": 10,
            "subsample": 1,
            "colsample_bytree": 1,
            "objective": "regression",
            "metric": "mae",
            "n_jobs": -1,  # Use all CPU cores
            "verbose": 2,
            "force_row_wise": True
        }
    }
}
```

#### Adjusting Group Filtering

In `data_loader.py`, modify the minimum hours threshold:

```python
class ClickHouseDataLoader:
    def __init__(self):
        self.min_hours = 24102  # Change this value
```

This filters out groups with fewer than the specified hours of data, ensuring model quality by excluding sparse groups.

### Training Process Details

1. **Data Extraction**:
   - Connects to ClickHouse
   - Fetches data in 500k row chunks
   - Filters by date range and optionally by specific groups

2. **Group Filtering**:
   - Calculates total hours per group (zone × borough × service_type)
   - Removes groups below `min_hours` threshold
   - Combines group columns into single `group_id` for memory efficiency

3. **Data Preprocessing**:
   - Merges `pickup_date` and `pickup_hour` into `pickup_datetime`
   - Interpolates missing hours per group using exponential weighted mean
   - Validates data structure

4. **Feature Engineering**:
   - Creates temporal features (hour, day of week, rush hour, etc.)
   - Generates lag features and rolling statistics
   - Applies cyclical encoding (sin/cos)
   - Calculates group statistics
   - Splits into train/test (last 8 months = test)

5. **Model Training**:
   - Encodes categorical features (time_of_day, group_id)
   - Trains LightGBM with early stopping on validation set
   - Evaluates performance (MAE on train and test)
   - Retrains on full dataset with best iteration count
   - Saves model, encoders, and artifacts

6. **Artifacts Saved**:
   - `models/LIGHTGBM_model.pkl`: Trained model
   - `models/encoders/category_mappings.pkl`: Categorical encodings
   - `models/pipeline_artifacts.pkl`: Feature names and config
   - `results/model_results.csv`: Performance metrics
   - `results/feature_importances/`: Feature importance rankings

### Performance Metrics

Example results:
- **Training MAE**: ~8-12 trips/hour
- **Test MAE**: ~8-12 trips/hour
- **60% improvement** over baseline (naive forecasting)

---

## 🖥 Streamlit Dashboard

Interactive web application for exploring historical demand and generating forecasts.

### Project Structure

```
demand_prediction/streamlit_app/
├── app.py                   # Main Streamlit application
├── appconfig.py             # Dashboard configuration
├── models.py                # Data models (ForecastGroup, ForecastResult)
├── database.py              # ClickHouse connection and queries
├── forecaster_manager.py    # Forecaster lifecycle management
├── data_processor.py        # Data transformation and statistics
├── visualization.py         # Altair chart creation
└── ui_components.py         # Reusable UI elements

results/                     # Forecast outputs and exports
```

### Running the Dashboard

```bash
# Standard run
streamlit run demand_prediction/streamlit_app/app.py

# Custom port
streamlit run demand_prediction/streamlit_app/app.py --server.port 8501

# Debug mode
streamlit run demand_prediction/streamlit_app/app.py --logger.level=debug
```

### Dashboard Features

#### 📍 Location Selection
- Browse available boroughs, zones, and service types
- Automatic filtering based on data availability
- Only shows groups with sufficient historical data (>24k hours)

#### 📅 Date Range Selection
- Choose start and end dates for historical data
- Default: 2025-08-01 to 2025-09-01
- Validates date ranges automatically

#### ⚙️ Forecast Configuration
- **Horizon**: 1-1500 hours into the future (default: 2 hours)
- **Model**: LightGBM (currently single model, extensible)
- Real-time progress tracking during forecast generation

#### 📊 Visualizations
- **Interactive Line Charts**: Historical vs forecasted demand with Altair
- **Confidence Intervals**: ±20% prediction bands
- **Hover Tooltips**: Detailed information on each data point
- **Zoom & Pan**: Interactive exploration

#### 📈 Statistics Dashboard
- Historical data summary (count, mean, std)
- Forecast summary (predictions, min, max)
- Percentage change in average demand
- Data preview (last 72 hours shown)

#### 💾 Export & Controls
- **CSV Export**: Download forecasts with timestamps
- **Cache Management**: Clear cached queries
- **Forecaster Restart**: Reload models without redeployment
- **System Status**: Monitor forecaster health

### Dashboard Configuration

Edit `streamlit_app/appconfig.py`:

```python
@dataclass
class AppConfig:
    PAGE_TITLE: str = "Taxi Demand Forecasting"
    PAGE_ICON: str = "🚕"
    LAYOUT: str = "wide"
    CACHE_TTL: int = 300  # Cache time-to-live (seconds)
    
    # Forecast settings
    MIN_FORECAST_HORIZON: int = 1
    MAX_FORECAST_HORIZON: int = 1500
    DEFAULT_FORECAST_HORIZON: int = 2
    
    # Data requirements
    MIN_RECORD_COUNT: int = 24102  # Minimum hours per group
    
    RESULTS_DIR: str = "results"
    LOG_LEVEL: str = "INFO"
```

---

## Installation

### Prerequisites
- Python 3.8+
- ClickHouse database with taxi data
- pip package manager

### Setup

```bash
# Clone repository
git clone https://github.com/Ziadashraf301/TaxiTrack.git
cd TAXITRACK/demand_prediction/streamlit_app

# Create virtual environment
python -m venv venv
source venv\Scripts\activate 

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
# Core ML
pandas>=2.0.0
numpy>=1.23.0
lightgbm>=3.3.0
scikit-learn>=1.2.0
joblib>=1.2.0

# Database
clickhouse-connect>=0.6.0

# Dashboard
streamlit>=1.28.0
altair>=5.0.0

# Utilities
python-dotenv>=1.0.0
```

---

## Quick Start

### 1. Configure Database Connection

Edit `model_dev/config.py`:

```python
CLICKHOUSE_CONFIG = {
    "host": "your-clickhouse-host",
    "port": 8123,
    "username": "your_username",
    "password": "your_password",
    "database": "data_warehouse"
}
```

### 2. Train Models

```bash
python demand_prediction/model_dev/run_pipeline.py
```

Expected output:
```
🚀 Starting Production Training Pipeline...
✅ Data loaded: 4,564,311 rows
✅ Groups after filtering: 76
✅ Feature engineering completed
LIGHTGBM: train_mae=9.34 - test_mae=12.42
✅ Pipeline completed successfully!
```

### 3. Test Forecaster

```bash
python demand_prediction/model_dev/test_forecaster.py
```

### 4. Launch Dashboard

```bash
streamlit run demand_prediction/streamlit_app/app.py
```

### 5. Generate Forecasts

1. Select borough, zone, and service type
2. Choose date range and forecast horizon
3. Click "Run Forecast"
4. View results and download CSV

---

## Configuration

### Key Configuration Files

| File | Purpose |
|------|---------|
| `model_dev/config.py` | Model, data, ClickHouse, and path configuration |
| `streamlit_app/appconfig.py` | Dashboard settings and UI parameters |

### Adjusting Model Parameters

To tune LightGBM performance, edit `MODEL_CONFIG` in `config.py`:

```python
MODEL_CONFIG = {
    "models": {
        "LIGHTGBM": {
            "n_estimators": 5000,      # Max trees
            "learning_rate": 0.05,     # Step size
            "num_leaves": 500,         # Complexity
            "min_child_samples": 10,   # Min samples per leaf
        }
    }
}
```

### Filtering Groups by Data Volume

In `data_loader.py`:

```python
def filter_low_volume_groups(self, df, min_hours=None):
    if min_hours is None:
        min_hours = self.min_hours  # Default: 24102 (75% of total hours between 2022 and 2025)
```

This ensures only groups with sufficient historical data are used for training, improving model reliability.

---

## Development

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | Central configuration management |
| `logger.py` | UTF-8 safe logging with file/console output |
| `data_loader.py` | ClickHouse data extraction and preprocessing |
| `feature_engineer.py` | Feature generation and train/test splitting |
| `encoder.py` | Categorical encoding for LightGBM |
| `pipeline.py` | Training orchestration and model persistence |
| `forecast.py` | Iterative multi-step forecasting engine |
| `database.py` | Dashboard database queries |
| `forecaster_manager.py` | Forecaster initialization and execution |
| `visualization.py` | Altair chart generation |
| `ui_components.py` | Reusable Streamlit widgets |

### Testing

```bash
# Test data loading
python demand_prediction/model_dev/test_pipeline.py

# Test forecasting
python demand_prediction/model_dev/test_forecaster.py

# Test dashboard components (manual)
streamlit run demand_prediction/streamlit_app/app.py
```
## 🎥 Demo Video

[▶️ Watch the System in Action](Demo.mp4)

---

## 📊 System Capabilities

- **Training Speed**: 5-10 minutes for 75+ groups
- **Prediction Latency**: <100ms per group
- **Scalability**: Handles millions of historical records
- **Accuracy**: 60% MAE improvement over baseline
- **Dashboard Load**: <5 seconds with caching

---

## 🙏 Acknowledgments

Built with:
- **LightGBM** for gradient boosting
- **Streamlit** for interactive dashboards
- **ClickHouse** for data warehousing
- **Altair** for visualizations

---
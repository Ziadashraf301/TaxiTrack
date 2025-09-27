# Data Configuration - REMOVED other numeric features
DATA_CONFIG = {
    "timestamp_col": "pickup_datetime",
    "target_col": "total_trips", 
    "group_cols": ["pickup_zone", "pickup_borough", "service_type"],
    "hour_col": "pickup_hour",
    
    # Feature columns - ONLY target remains
    "numeric_features": [],  # EMPTY - only using target variable
    "categorical_features": ["pickup_zone", "pickup_borough", "service_type"],
    
    # Time series configuration
    "frequency": "h",
    "forecast_horizon": 1,
    "test_days": 7,
    
    # Feature engineering
    "lag_features": [1, 2, 3, 4, 5, 6, 12, 24, 168],
    "rolling_windows": [3, 6, 12, 24],
    "seasonal_periods": {
        "hourly": 24,
        "daily": 7,
        "weekly": 52
    },
    
    # New settings
    "min_date": "2019-01-01",
    "max_date": "2020-04-30", 
    "interpolation_method": "linear",
    "fillna_value": 0
}

# Model Configuration - Enhanced with time series specific settings
MODEL_CONFIG = {
    "models": {
        "xgboost": {
            "n_estimators": 3000, 
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "device": "cuda",
            "verbosity": 1  
        },
        "decision_tree": {
            "max_depth": 15, 
            "random_state": 42
        }
    },
 
    # Feature importance settings
    "calculate_feature_importance": True,
    "top_features": 20
}

# Path Configuration - Updated for our project structure
PATH_CONFIG = {
    "base_dir": "./",
    "models_dir": "./models/",
    "results_dir": "./results/",
    "logs_dir": "./logs/",
    "encoder_dir": "./models/encoders/",
    # Streamlit app paths
    "streamlit_data": "./streamlit_app/data/",
    "streamlit_models": "./streamlit_app/models/",
    
    # Specific file names
    "pipeline_artifacts": "pipeline_artifacts.pkl",
    "model_results": "model_results.csv",
    "feature_importance": "feature_importance.csv"
}

# ClickHouse Configuration
CLICKHOUSE_CONFIG = {
    "host": "clickhouse",
    "port": 8123,
    "username": "ziadashraf98765", 
    "password": "x5x6x7x8",
    "database": "data_warehouse",
    "chunk_size": 500000,
    "default_table": "mart_demand_prediction"  # You'll need to specify your actual table name
}

# Feature Engineering Configuration - More detailed
FEATURE_CONFIG = {
    "time_features": [
        "hour", "dayofweek", "dayofmonth", "weekofyear", "month", "year",
        "is_weekend", "is_night", "is_rush_hour"
    ],
    "cyclic_features": {
        "hour": 24,
        "dayofweek": 7,
        "month": 12
    },
    "lag_features": {
        "hourly": [1, 2, 3, 4, 5, 6, 12, 24],
        "daily": [24, 48, 72],
        "weekly": [168, 336]
    },
    "rolling_windows": {
        "short_term": [3, 6, 12],    # hours
        "medium_term": [24, 48],     # 1-2 days
        "long_term": [168]           # 1 week
    },
    "statistical_features": ["mean", "std", "min", "max", "ema"]
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "metrics": {
        "mae": "Mean Absolute Error",
        "rmse": "Root Mean Squared Error", 
        "mape": "Mean Absolute Percentage Error",
        "smape": "Symmetric Mean Absolute Percentage Error"
    },
    "benchmark_models": ["naive_seasonal", "last_value", "average"],
    "confidence_intervals": True,
    "confidence_level": 0.95
}

# Production Configuration
PRODUCTION_CONFIG = {
    "retraining_schedule": "weekly",  # Options: daily, weekly, monthly
    "monitoring_metrics": ["data_drift", "concept_drift", "prediction_drift"],
    "alert_thresholds": {
        "mae_increase": 0.2,  # 20% increase in MAE triggers alert
        "data_drift": 0.1,    # 10% drift threshold
        "prediction_delay": 3600  # 1 hour max delay
    },
    "model_refresh_days": 30
}
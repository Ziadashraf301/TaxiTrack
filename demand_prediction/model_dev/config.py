# Data Configuration - REMOVED other numeric features
DATA_CONFIG = {
    "timestamp_col": "pickup_datetime",
    "target_col": "total_trips", 
    "group_cols": ["pickup_zone", "pickup_borough", "service_type"],
    "group_col":  "group_id",
    "hour_col": "pickup_hour",
    
    # Feature columns
    "numeric_features": [], 
    "categorical_features": ["time_of_day", "group_id"],
    
    # Time series configuration
    "frequency": "h",
    "test_months": 8,
    
    # Feature engineering
    "lag_features": [24, 168],
    "rolling_windows": [3, 6, 9, 12, 15, 18 , 21 , 24, 168]
}

# Model Configuration - Enhanced with time series specific settings
MODEL_CONFIG = {
    "models": {
            "LIGHTGBM": {

            # Core parameters
            "n_estimators": 5000,
            "learning_rate": 0.05,
            "num_leaves": 500,
            "min_child_samples": 10,
            "subsample": 1,
            "colsample_bytree": 1,
            "random_state": 42,
            
            # Parallel processing (CPU )
            "device": "cpu",
            "n_jobs": -1,  # Use all CPU cores (-1 means all available)

            "categorical_feature": ['time_of_day','group_id'],

            # Performance optimization
            "objective": "regression",  # Change based on your task
            "metric": "mae",
            "verbose": 2,
            
            # Memory optimization
            "feature_pre_filter": False,
            
            # Additional speed optimizations
            "force_row_wise": True,  # Faster for datasets with many rows (comment out force_col_wise)
        
            }
    }
}   

# Path Configuration - Updated for our project structure
PATH_CONFIG = {
    "models_dir": "./models/",
    "results_dir": "./results/",
    "logs_dir": "./logs/",
    "encoder_dir": "./models/encoders/",

    # Specific file names
    "pipeline_artifacts": "pipeline_artifacts.pkl",
    "model_results": "model_results.csv",
    "feature_importance": "feature_importance.csv"
}

# ClickHouse Configuration
CLICKHOUSE_CONFIG = {
    "host": "localhost",
    "port": 8123,
    "username": "ziadashraf98765", 
    "password": "x5x6x7x8",
    "database": "data_warehouse"
}

TABLE_CONFIG = {"table" : "mart_demand_prediction",
                "start_date" : "2022-01-01",
                "end_date" : "2025-08-31"}
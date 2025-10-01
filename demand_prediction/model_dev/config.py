# Data Configuration - REMOVED other numeric features
DATA_CONFIG = {
    "timestamp_col": "pickup_datetime",
    "target_col": "total_trips", 
    "group_cols": ["pickup_zone", "pickup_borough", "service_type"],
    "hour_col": "pickup_hour",
    
    # Feature columns
    "numeric_features": [], 
    "categorical_features": ["pickup_zone", "pickup_borough", "service_type"],
    
    # Time series configuration
    "frequency": "h",
    "test_months": 3,
    
    # Feature engineering
    "lag_features": [1, 2, 3, 24, 168],
    "rolling_windows": [3, 6, 12, 24, 168],
    "seasonal_periods": {
        "hourly": 24,
        "daily": 7,
        "weekly": 52
        
    }
}

# Model Configuration - Enhanced with time series specific settings
MODEL_CONFIG = {
    "models": {
            "xgboost": {
                "n_estimators": 10000, 
                "max_depth": 8,
                "learning_rate": 0.1,
                "random_state": 42,
                "subsample": 0.8,
                "colsample_bytree": 0.4,
                "tree_method":"hist", 
                "device":"cuda",
                "verbosity": 2,
                "early_stopping_rounds": 100,
                "reg_lambda": 50,
                "reg_alpha": 20,
            },
            "decision_tree": {
                "max_depth": 15,                
                "ccp_alpha": 0.001,          
                "random_state": 42
            },
            "LIGHTGBM": {
                    "n_estimators": 500,
                    "learning_rate": 0.05,
                    "max_depth": 8,
                    "num_leaves": 31,
                    "min_child_samples": 20,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "random_state": 42
            },
            "ridge": {
                    "alpha": 1.0
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
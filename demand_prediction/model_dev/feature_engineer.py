import os
import joblib
import pandas as pd
import numpy as np
from config import DATA_CONFIG, PATH_CONFIG
from logger import setup_logger
from pathlib import Path

log_dir = Path(PATH_CONFIG["logs_dir"])
logger = setup_logger(__name__, f"{log_dir}/pipeline.log")

class TimeSeriesFeatureEngineer:
    def __init__(self):
        self.config = DATA_CONFIG
        self.encoder_dir = PATH_CONFIG["encoder_dir"]
        self.Training = True
        self.Split = True
        self._is_fitted = False

    def fit(self, X):
        """Fit the feature engineer on training data"""
        logger.info("Fitting TimeSeriesFeatureEngineer on %d rows", len(X))
        self.Training = True
        self._is_fitted = True
        return self

    def transform(self, df):
        """Transform data with zone-aware features"""
        logger.info("Starting feature engineering with %d rows", len(df))
        engineered_data = self._engineer_features(df)
        X, y, timestamps = self._prepare_features_target(engineered_data)

        if self.Split:
            X_train, X_test, y_train, y_test, train_times, test_times = self._group_aware_time_split(X, y, timestamps)
            logger.info("✅ Feature engineering completed (training mode)")
            return X_train, X_test, y_train, y_test, train_times, test_times

        logger.info("✅ Feature engineering completed (inference mode)")
        return X, y, timestamps

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def _engineer_features(self, df):
        df = df.copy()
        ts, tgt, groups = self.config["timestamp_col"], self.config["target_col"], self.config["group_col"]

        df = df.sort_values([groups , ts]).reset_index(drop=True)

        df = self._create_time_features(df, ts)
        df = self._create_lag_features(df, tgt, groups)
        df = self._create_rolling_features(df, tgt, groups)
        df = self._create_seasonal_features(df)
        df = self._handle_missing_values(df, tgt, groups)
        return df


    def _prepare_features_target(self, df):
        """Prepare features and target variable - NO NaN HANDLING"""
        timestamp_col = self.config["timestamp_col"]
        target_col = self.config["target_col"]
        
        # Feature columns - exclude Dates, timestamp, and target
        exclude_cols = [timestamp_col, target_col, "pickup_date", "pickup_hour"] #, "pickup_zone"
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        if not feature_cols:
            logger.error("❌ No feature columns left after exclusion!")
            raise ValueError("No feature columns for training")

        # Get features, target, timestamps, and groups
        X = df[feature_cols]
        y = df[target_col]
        timestamps = df[timestamp_col]
        
        # Ensure all arrays have the same length
        assert len(X) == len(y) == len(timestamps), \
            f"Length mismatch: X={len(X)}, y={len(y)}, timestamps={len(timestamps)}"
        
        logger.info(f"📊 Final dataset - X: {X.shape}, y: {len(y)}, timestamps: {len(timestamps)}")
        
        return X, y, timestamps


    def _group_aware_time_split(self, X, y, timestamps):
        """Split each month: train on all but last 3 months, test on last 3 months"""
           
        # Ensure timestamps are datetime
        timestamps = pd.to_datetime(timestamps)

        # Get month boundaries
        month_end = timestamps.max().normalize()
        test_months = DATA_CONFIG['test_months']

        # Cutoff 
        cutoff_date = month_end - pd.DateOffset(months=test_months)

        # Train = everything before or equal cutoff
        train_mask = timestamps <= cutoff_date
        test_mask = timestamps > cutoff_date

        # Apply masks
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        train_times, test_times = timestamps[train_mask], timestamps[test_mask]

        # Logging
        logger.info(f"Train set: {len(X_train)} samples ({train_times.min()} → {train_times.max()})")
        logger.info(f"Test set: {len(X_test)} samples ({test_times.min()} → {test_times.max()})")

        return X_train, X_test, y_train, y_test, train_times, test_times


    def _create_time_features(self, df, timestamp_col):
        logger.info("Creating time features")
        df['hour'] = df[timestamp_col].dt.hour
        df['dayofweek'] = df[timestamp_col].dt.dayofweek
        df['dayofmonth'] = df[timestamp_col].dt.day
        df['is_weekend'] = df['dayofweek'].isin([5, 6])
        df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) |
                              ((df['hour'] >= 16) & (df['hour'] <= 19)))

        # Time-of-day categories (morning: 0, midday: 1, evening: 2, night: 3)
        def time_of_day( hour):
                if 6 <= hour < 10:
                    return "morning"
                elif 10 <= hour < 16:
                    return "midday"
                elif 16 <= hour < 21:
                    return "evening"
                else:
                    return "night"

        df['time_of_day'] = df['hour'].apply(time_of_day)

        return df


    def _create_lag_features(self, df, target_col, group_col):
        logger.info("Creating lag features: %s", self.config["lag_features"])
        
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found")
            return df
        
        for lag in self.config["lag_features"]:
            col_name = f'lag_{lag}h'
            df[col_name] = df.groupby(group_col)[target_col].shift(lag)
            
            if not self.Training:
                df[col_name] = df.groupby(group_col)[col_name].ffill()
                if df[col_name].isna().any():
                    group_means = df.groupby(group_col)[target_col].transform('mean')
                    df[col_name] = df[col_name].fillna(group_means).fillna(0)
                    
        return df


    def _create_rolling_features(self, df, target_col, group_col):
        """Create rolling window features with proper handling of missing values"""
        logger.info("Creating rolling features: %s", self.config["rolling_windows"])
        
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found, skipping rolling features")
            return df
        
        for window in self.config["rolling_windows"]:
            # Rolling mean
            col_name = f'rolling_mean_{window}h'
            df[col_name] = df.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())            
            
            # Exponential moving average
            col_name = f'ema_{window}h'
            df[col_name] = df.groupby(group_col)[target_col].transform(
                lambda x: x.ewm(span=window, min_periods=1).mean())
            
        return df


    def _create_seasonal_features(self, df):
        logger.info("Creating seasonal features")
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        return df


    def _handle_missing_values(self, df, target_col, group_col):
        logger.info("Handling missing values")
        
        exclude_cols = [target_col, self.config["timestamp_col"], "time_of_day" + group_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        for col in feature_cols:
            if df[col].isna().any():
                if 'lag' in col or 'rolling' in col or 'ema' in col:
                    df[col] = df.groupby(group_col)[col].ffill()
                df[col] = df[col].fillna(0)
        
        return df

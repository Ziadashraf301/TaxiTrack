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
        self.zone_stats = {}

    def fit(self, X):
        """Fit the feature engineer on training data"""
        logger.info("Fitting TimeSeriesFeatureEngineer on %d rows", len(X))
        self.Training = True
        self._is_fitted = True
        return self

    def transform(self, df):
        """Transform data with zone-aware features"""
        logger.info("Starting zone-aware feature engineering with %d rows", len(df))
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
        ts, tgt, groups = self.config["timestamp_col"], self.config["target_col"], self.config["group_cols"]

        df[ts] = pd.to_datetime(df[ts])
        df = df.sort_values(groups + [ts]).reset_index(drop=True)

        df = self._create_time_features(df, ts)
        df = self._create_lag_features(df, tgt, groups)
        df = self._create_rolling_features(df, tgt, groups)
        df = self._create_seasonal_features(df)

        # remove not tomorow
        if self.Training:
            if not self.zone_stats:  # if not already set
                self.zone_stats = self._compute_zone_statistics(df, tgt)
        else:
            self._load_artifacts()

        df = self._create_zone_features(df)
        df = self._create_zone_interaction_features(df)
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


    def _create_lag_features(self, df, target_col, group_cols):
        logger.info("Creating lag features: %s", self.config["lag_features"])
        
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found")
            return df
        
        for lag in self.config["lag_features"]:
            col_name = f'lag_{lag}h'
            df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
            
            if not self.Training:
                df[col_name] = df.groupby(group_cols)[col_name].ffill()
                if df[col_name].isna().any():
                    group_means = df.groupby(group_cols)[target_col].transform('mean')
                    df[col_name] = df[col_name].fillna(group_means).fillna(0)
                    
        return df


    def _create_rolling_features(self, df, target_col, group_cols):
        """Create rolling window features with proper handling of missing values"""
        logger.info("Creating rolling features: %s", self.config["rolling_windows"])
        
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found, skipping rolling features")
            return df
        
        for window in self.config["rolling_windows"]:
            # Rolling mean
            col_name = f'rolling_mean_{window}h'
            df[col_name] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())            
            
            # Exponential moving average
            col_name = f'ema_{window}h'
            df[col_name] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.ewm(span=window, min_periods=1).mean())
            
        return df


    def _create_seasonal_features(self, df):
        logger.info("Creating seasonal features")
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        return df


    def _compute_zone_statistics(self, df, target_col):
        """Compute comprehensive per-zone statistics"""
        logger.info("Creating zone statistics")

        zone_stats = {}
        
        for zone in df['pickup_zone'].unique():
            zone_data = df[df['pickup_zone'] == zone].copy()
            zone_trips = zone_data[target_col]
            
            if len(zone_trips) > 0:
                # Basic statistics
                zone_stats[zone] = {
                    'mean': float(zone_trips.mean()),
                    'median': float(zone_trips.median()),
                    'cv': float(zone_trips.std() / (zone_trips.mean() + 1)),
                }
                

                hourly_avg = zone_data.groupby('hour')[target_col].mean()
                zone_stats[zone]['peak_hour'] = int(hourly_avg.idxmax())
                zone_stats[zone]['off_peak_hour'] = int(hourly_avg.idxmin())
                zone_stats[zone]['peak_demand'] = float(hourly_avg.max())
                zone_stats[zone]['off_peak_demand'] = float(hourly_avg.min())
                zone_stats[zone]['hour_variance'] = float(hourly_avg.std())
                
                # Weekend patterns
                weekend_mean = zone_data[zone_data['is_weekend']][target_col].mean()
                weekday_mean = zone_data[~zone_data['is_weekend']][target_col].mean()
                zone_stats[zone]['weekend_ratio'] = float(weekend_mean / (weekday_mean + 1))
                zone_stats[zone]['weekend_mean'] = float(weekend_mean)
                zone_stats[zone]['weekday_mean'] = float(weekday_mean)
                

                rush_mean = zone_data[zone_data['is_rush_hour']][target_col].mean()
                non_rush_mean = zone_data[~zone_data['is_rush_hour']][target_col].mean()
                zone_stats[zone]['rush_hour_ratio'] = float(rush_mean / (non_rush_mean + 1))
        
        logger.info(f"Computed statistics for {len(zone_stats)} zones")
        
        joblib.dump(zone_stats, os.path.join(self.encoder_dir, "zone_stats.pkl"))
        logger.info("Saved Zone Statistics to %s", self.encoder_dir)
        
        return zone_stats


    def _create_zone_features(self, df):
        """Create zone-based features using pre-computed statistics"""
        logger.info("Creating zone features from statistics")
              
        # Zone statistics
        for stat in ['mean', 'cv', 'median', 'peak_hour', 'off_peak_hour',
                     'peak_demand', 'off_peak_demand', 'hour_variance',
                     'weekend_ratio', 'weekend_mean', 'weekday_mean', 'rush_hour_ratio']:
            df[f'zone_{stat}'] = df['pickup_zone'].map(
                lambda z: self.zone_stats.get(z, {}).get(stat, 0)
            )
        
        return df


    def _create_zone_interaction_features(self, df):
        """Create interaction features between time and zone patterns"""
        logger.info("Creating zone interaction features")
        
        # Is current hour the peak hour for this zone?
        df['is_zone_peak_hour'] = (df['hour'] == df['zone_peak_hour']).astype(int)
        df['is_zone_off_peak_hour'] = (df['hour'] == df['zone_off_peak_hour']).astype(int)
        
        # Weekend interaction
        df['weekend_zone_effect'] = df['is_weekend'] * df['zone_weekend_ratio']
        
        # Rush hour interaction
        df['rush_zone_effect'] = df['is_rush_hour'] * df['zone_rush_hour_ratio']
        
        # Hour similarity to zone pattern
        df['hour_sin_x_peak'] = df['hour_sin'] * df['zone_peak_demand']
        df['hour_cos_x_peak'] = df['hour_cos'] * df['zone_peak_demand']
        
        # Volatility interaction
        df['time_volatility'] = df['is_rush_hour'] * df['zone_cv']
        
        return df


    def _handle_missing_values(self, df, target_col, group_cols):
        logger.info("Handling missing values")
        
        exclude_cols = [target_col, self.config["timestamp_col"], "pickup_date", "pickup_hour", "time_of_day"] + group_cols
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        for col in feature_cols:
            if df[col].isna().any():
                if 'lag' in col or 'rolling' in col or 'ema' in col:
                    df[col] = df.groupby(group_cols)[col].ffill()
                df[col] = df[col].fillna(0)
        
        return df


    def _load_artifacts(self):
        """Load saved artifacts in inference mode"""
        if not self.zone_stats:
            zone_stats_path = os.path.join(self.encoder_dir, "zone_stats.pkl")
            if os.path.exists(zone_stats_path):
                self.zone_stats = joblib.load(zone_stats_path)
                logger.info("Loaded zone statistics")
    


    def _is_pickup_borough_Manhattan(self, df):

        logger.info(" Add a binary column 'is_pickup_manhattan' and drop the original 'pickup_borough' column")

        pickup_col = "pickup_borough"

        # Create binary column
        df["is_pickup_manhattan"] = (df[pickup_col] == "Manhattan").astype(int)
        # Drop the original column
        df = df.drop(columns=[pickup_col])
        
        return df

import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from config import DATA_CONFIG, PATH_CONFIG
from logger import setup_logger
from pathlib import Path

log_dir = Path(PATH_CONFIG["logs_dir"])
logger = setup_logger(__name__, f"{log_dir}/pipeline.log")


class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.config = DATA_CONFIG
        self.feature_names = []
        self.encoder_dir = PATH_CONFIG["encoder_dir"]
        self.Training = True
        self.preprocessor = None
        self.encoded_feature_names = []
        self._is_fitted = False
        self.zone_stats = {}
        self.zone_target_map = {}

    def fit(self, X, y=None):
        """Fit the feature engineer on training data"""
        logger.info("Fitting TimeSeriesFeatureEngineer on %d rows", len(X))
        
        X = X.copy()
        timestamp_col = self.config["timestamp_col"]
        group_cols = self.config["group_cols"]
        target_col = self.config["target_col"]
        
        X[timestamp_col] = pd.to_datetime(X[timestamp_col])
        X = X.sort_values(group_cols + [timestamp_col])
        
        # Compute zone statistics and target encoding
        logger.info("Computing per-zone statistics for 262 zones...")
        self.zone_stats = self._compute_zone_statistics(X, target_col, group_cols)
        self.zone_target_map = self._compute_target_encoding(X, target_col)
        
        # OneHotEncode only low-cardinality categoricals (service_type, borough)
        os.makedirs(self.encoder_dir, exist_ok=True)
        
        logger.info("Fitting encoders for service_type and pickup_borough only")
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("service", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ['service_type']),
                ("borough", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ['pickup_borough'])
            ],
            remainder="drop"
        )
        
        self.preprocessor.fit(X[['service_type', 'pickup_borough']])
        
        service_features = self.preprocessor.named_transformers_['service'].get_feature_names_out(['service_type'])
        borough_features = self.preprocessor.named_transformers_['borough'].get_feature_names_out(['pickup_borough'])
        self.encoded_feature_names = list(np.concatenate([service_features, borough_features]))
        
        # Save all artifacts
        joblib.dump(self.preprocessor, os.path.join(self.encoder_dir, "categorical_encoder.pkl"))
        joblib.dump(self.zone_stats, os.path.join(self.encoder_dir, "zone_stats.pkl"))
        joblib.dump(self.zone_target_map, os.path.join(self.encoder_dir, "zone_target_map.pkl"))
        logger.info("Saved encoders and zone artifacts to %s", self.encoder_dir)
        
        self._is_fitted = True
        return self

    def _compute_zone_statistics(self, df, target_col, group_cols):
        """Compute comprehensive per-zone statistics"""
        zone_stats = {}
        
        for zone in df['pickup_zone'].unique():
            zone_data = df[df['pickup_zone'] == zone].copy()
            zone_trips = zone_data[target_col]
            
            if len(zone_trips) > 0:
                # Basic statistics
                zone_stats[zone] = {
                    'mean': float(zone_trips.mean()),
                    'std': float(zone_trips.std()),
                    'median': float(zone_trips.median()),
                    'q25': float(zone_trips.quantile(0.25)),
                    'q75': float(zone_trips.quantile(0.75)),
                    'max': float(zone_trips.max()),
                    'min': float(zone_trips.min()),
                    'cv': float(zone_trips.std() / (zone_trips.mean() + 1)),
                }
                
                # Temporal patterns
                zone_data['hour'] = zone_data[self.config["timestamp_col"]].dt.hour
                zone_data['dayofweek'] = zone_data[self.config["timestamp_col"]].dt.dayofweek
                
                hourly_avg = zone_data.groupby('hour')[target_col].mean()
                zone_stats[zone]['peak_hour'] = int(hourly_avg.idxmax())
                zone_stats[zone]['off_peak_hour'] = int(hourly_avg.idxmin())
                zone_stats[zone]['peak_demand'] = float(hourly_avg.max())
                zone_stats[zone]['off_peak_demand'] = float(hourly_avg.min())
                zone_stats[zone]['hour_variance'] = float(hourly_avg.std())
                
                # Weekend patterns
                zone_data['is_weekend'] = zone_data['dayofweek'].isin([5, 6])
                weekend_mean = zone_data[zone_data['is_weekend']][target_col].mean()
                weekday_mean = zone_data[~zone_data['is_weekend']][target_col].mean()
                zone_stats[zone]['weekend_ratio'] = float(weekend_mean / (weekday_mean + 1))
                zone_stats[zone]['weekend_mean'] = float(weekend_mean)
                zone_stats[zone]['weekday_mean'] = float(weekday_mean)
                
                # Rush hour patterns
                zone_data['is_rush_hour'] = (
                    ((zone_data['hour'] >= 7) & (zone_data['hour'] <= 9)) |
                    ((zone_data['hour'] >= 16) & (zone_data['hour'] <= 19))
                )
                rush_mean = zone_data[zone_data['is_rush_hour']][target_col].mean()
                non_rush_mean = zone_data[~zone_data['is_rush_hour']][target_col].mean()
                zone_stats[zone]['rush_hour_ratio'] = float(rush_mean / (non_rush_mean + 1))
        
        logger.info(f"Computed statistics for {len(zone_stats)} zones")
        return zone_stats

    def _compute_target_encoding(self, df, target_col):
        """Target encoding: map each zone to its historical average demand"""
        zone_target_map = df.groupby('pickup_zone')[target_col].mean().to_dict()
        
        # Add smoothing for rare zones
        global_mean = df[target_col].mean()
        min_samples = 100
        
        for zone in zone_target_map.keys():
            zone_count = len(df[df['pickup_zone'] == zone])
            if zone_count < min_samples:
                # Smooth with global mean
                weight = zone_count / min_samples
                zone_target_map[zone] = (
                    weight * zone_target_map[zone] + 
                    (1 - weight) * global_mean
                )
        
        logger.info(f"Computed target encoding for {len(zone_target_map)} zones")
        return zone_target_map

    def transform(self, df):
        """Transform data with zone-aware features"""
        logger.info("Starting zone-aware feature engineering with %d rows", len(df))
        df = df.copy()
        timestamp_col = self.config["timestamp_col"]
        target_col = self.config["target_col"]
        group_cols = self.config["group_cols"]

        required_cols = [timestamp_col] + group_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(group_cols + [timestamp_col]).reset_index(drop=True)

        # Load artifacts if in inference mode
        if not self.Training:
            self._load_artifacts()

        # Core features
        df = self._create_time_features(df, timestamp_col)
        df = self._create_lag_features(df, target_col, group_cols)
        df = self._create_rolling_features(df, target_col, group_cols)
        df = self._create_seasonal_features(df)
        
        # Zone features
        df = self._create_zone_features(df)
        df = self._create_zone_interaction_features(df)

        # Encode low-cardinality categoricals only
        df = self._encode_categorical_features(df)
        
        df = self._handle_missing_values(df, target_col, group_cols)

        if self.Training:

            # Prepare data for modeling
            X, y, timestamps, groups = self._prepare_features_target(df)

            # Delete to free memory
            del df
            gc.collect()
            logger.debug("🧹 GC: Deleted raw df after feature engineering")

            # Time-based split ensuring equal time periods for all groups
            X_train, X_test, y_train, y_test, train_times, test_times = self._group_aware_time_split(X, y, timestamps, groups)


            del X, y, timestamps, groups
            gc.collect()
            logger.debug("🧹 GC: Deleted full dataset after time split")

            logger.info("✅ Feature engineering completed")

            return X_train, X_test, y_train, y_test, train_times, test_times

        logger.info("✅ Feature engineering completed")
        return df
    

    def _load_artifacts(self):
        """Load saved artifacts in inference mode"""
        if not self.zone_stats:
            zone_stats_path = os.path.join(self.encoder_dir, "zone_stats.pkl")
            if os.path.exists(zone_stats_path):
                self.zone_stats = joblib.load(zone_stats_path)
                logger.info("Loaded zone statistics")
        
        if not self.zone_target_map:
            zone_target_path = os.path.join(self.encoder_dir, "zone_target_map.pkl")
            if os.path.exists(zone_target_path):
                self.zone_target_map = joblib.load(zone_target_path)
                logger.info("Loaded zone target encoding")

    def fit_transform(self, X, y=None):
        self.Training = True
        self.fit(X, y)
        return self.transform(X)


    def _create_time_features(self, df, timestamp_col):
        logger.info("Creating time features")
        df['hour'] = df[timestamp_col].dt.hour
        df['dayofweek'] = df[timestamp_col].dt.dayofweek
        df['dayofmonth'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['weekofyear'] = df[timestamp_col].dt.isocalendar().week.astype(int)
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
        df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) |
                              ((df['hour'] >= 16) & (df['hour'] <= 19))).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['is_weekend'] == 0)).astype(int)

        # Time-of-day categories (morning: 0, midday: 1, evening: 2, night: 3)
        def time_of_day( hour):
                if 6 <= hour < 10:
                    return 0
                elif 10 <= hour < 16:
                    return 1
                elif 16 <= hour < 21:
                    return 2
                else:
                    return 3

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
            
            # Rolling std
            col_name = f'rolling_std_{window}h'
            df[col_name] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std())
            
            # Rolling min
            col_name = f'rolling_min_{window}h'
            df[col_name] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).min())
            
            # Rolling max
            col_name = f'rolling_max_{window}h'
            df[col_name] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).max())
            
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
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df

    def _create_zone_features(self, df):
        """Create zone-based features using pre-computed statistics"""
        logger.info("Creating zone features from statistics")
        
        # Target encoding
        df['zone_target_encoded'] = df['pickup_zone'].map(self.zone_target_map).fillna(0)
        
        # Zone statistics
        for stat in ['mean', 'std', 'cv', 'median', 'peak_hour', 'off_peak_hour',
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

    def _encode_categorical_features(self, df):
        """Encode only service_type and pickup_borough (low cardinality)"""
        logger.info("Encoding categorical features: service_type, pickup_borough")

        if self.Training and self.preprocessor is None:
            logger.warning("Preprocessor not fitted, fitting now")
            os.makedirs(self.encoder_dir, exist_ok=True)
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("service", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ['service_type']),
                    ("borough", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ['pickup_borough'])
                ],
                remainder="drop"
            )
            
            encoded = self.preprocessor.fit_transform(df[['service_type', 'pickup_borough', 'time_of_day']])
            
            service_features = self.preprocessor.named_transformers_['service'].get_feature_names_out(['service_type'])
            borough_features = self.preprocessor.named_transformers_['borough'].get_feature_names_out(['pickup_borough'])
            self.encoded_feature_names = list(np.concatenate([service_features, borough_features]))
            
            joblib.dump(self.preprocessor, os.path.join(self.encoder_dir, "categorical_encoder.pkl"))
            
        elif not self.Training:
            if self.preprocessor is None:
                encoder_path = os.path.join(self.encoder_dir, "categorical_encoder.pkl")
                if not os.path.exists(encoder_path):
                    raise FileNotFoundError(f"Encoder not found at {encoder_path}")
                self.preprocessor = joblib.load(encoder_path)
                
                service_features = self.preprocessor.named_transformers_['service'].get_feature_names_out(['service_type'])
                borough_features = self.preprocessor.named_transformers_['borough'].get_feature_names_out(['pickup_borough'])
                self.encoded_feature_names = list(np.concatenate([service_features, borough_features]))
            
            encoded = self.preprocessor.transform(df[['service_type', 'pickup_borough']])
        else:
            encoded = self.preprocessor.transform(df[['service_type', 'pickup_borough']])

        encoded_df = pd.DataFrame(encoded, columns=self.encoded_feature_names, index=df.index)
        
        df = df.reset_index(drop=True)
        encoded_df = encoded_df.reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)
        
        return df

    def _handle_missing_values(self, df, target_col, group_cols):
        logger.info("Handling missing values")
        
        exclude_cols = [target_col, self.config["timestamp_col"], 'pickup_date', 'pickup_hour'] + group_cols
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        for col in feature_cols:
            if df[col].isna().any():
                if 'lag' in col or 'rolling' in col or 'ema' in col:
                    df[col] = df.groupby(group_cols)[col].ffill()
                df[col] = df[col].fillna(0)
        
        return df
    
    def _prepare_features_target(self, df):
        """Prepare features and target variable - NO NaN HANDLING"""
        timestamp_col = self.config["timestamp_col"]
        target_col = self.config["target_col"]
        group_cols = self.config["group_cols"]
        
        # Feature columns - exclude timestamp, target, and group columns
        exclude_cols = [timestamp_col, target_col, "pickup_date", "pickup_hour"] + group_cols
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        if not feature_cols:
            logger.error("❌ No feature columns left after exclusion!")
            raise ValueError("No feature columns for training")

        # Get features, target, timestamps, and groups
        X = df[feature_cols]
        y = df[target_col]
        timestamps = df[timestamp_col]
        groups = df[group_cols].astype(str).agg('_'.join, axis=1)
        
        # Ensure all arrays have the same length
        assert len(X) == len(y) == len(timestamps) == len(groups), \
            f"Length mismatch: X={len(X)}, y={len(y)}, timestamps={len(timestamps)}, groups={len(groups)}"
        
        logger.info(f"📊 Final dataset - X: {X.shape}, y: {len(y)}, timestamps: {len(timestamps)}, groups: {groups.nunique()}")
        
        return X, y, timestamps, groups


    def _group_aware_time_split(self, X, y, timestamps, groups):
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
        train_groups, test_groups = groups[train_mask], groups[test_mask]

        # Logging
        logger.info(f"Train set: {len(X_train)} samples ({train_times.min()} → {train_times.max()})")
        logger.info(f"Test set: {len(X_test)} samples ({test_times.min()} → {test_times.max()})")
        logger.info(f"Train groups: {train_groups.nunique()}, Test groups: {test_groups.nunique()}")

        return X_train, X_test, y_train, y_test, train_times, test_times
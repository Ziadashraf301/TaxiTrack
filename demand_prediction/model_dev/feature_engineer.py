import os
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from config import DATA_CONFIG, PATH_CONFIG



class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.config = DATA_CONFIG
        self.feature_names = []
        self.encoder_dir = PATH_CONFIG["encoder_dir"]
        self.Training = True  # Flag to differentiate between training and inference
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, df):
        """Create time-based features, lags, and rolling statistics using ONLY total_trips"""
        df = df.copy()
        timestamp_col = self.config["timestamp_col"]
        target_col = self.config["target_col"]
        group_cols = self.config["group_cols"]
        
        
        # Ensure datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by groups and time
        df = df.sort_values(group_cols + [timestamp_col])
        
        # Time-based features
        df = self._create_time_features(df, timestamp_col)
        
        # Lag features (ONLY using total_trips)
        df = self._create_lag_features(df, target_col, group_cols)
        
        # Rolling statistics (ONLY using total_trips)
        df = self._create_rolling_features(df, target_col, group_cols)
        
        # Seasonal features
        df = self._create_seasonal_features(df)
        
        # Encode categorical variables
        df = self._encode_categorical_features(df, Training=self.Training)
        
        return df
    
    def _create_time_features(self, df, timestamp_col):
        """Create basic time features"""
        df['hour'] = df[timestamp_col].dt.hour
        df['dayofweek'] = df[timestamp_col].dt.dayofweek
        df['dayofmonth'] = df[timestamp_col].dt.day
        df['weekofyear'] = df[timestamp_col].dt.isocalendar().week
        df['month'] = df[timestamp_col].dt.month
        df['year'] = df[timestamp_col].dt.year
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 16) & (df['hour'] <= 19))
        df['is_rush_hour'] = df['is_rush_hour'].astype(int)
        
        return df
    
    def _create_lag_features(self, df, target_col, group_cols):
        """Create lag features using ONLY total_trips"""
        for lag in self.config["lag_features"]:
            df[f'lag_{lag}h'] = df.groupby(group_cols)[target_col].shift(lag)
        
        # Same hour from previous days
        for days_back in [1, 2, 7]:
            df[f'lag_same_hour_{days_back}d'] = df.groupby(group_cols + ['hour'])[target_col].shift(days_back * 24)
        
        return df
    
    def _create_rolling_features(self, df, target_col, group_cols):
        """Create rolling statistics using ONLY total_trips"""
        for window in self.config["rolling_windows"]:
            # Basic rolling stats
            df[f'rolling_mean_{window}h'] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}h'] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f'rolling_min_{window}h'] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            df[f'rolling_max_{window}h'] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            
            # Exponential moving average
            df[f'ema_{window}h'] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.ewm(span=window, min_periods=1).mean()
            )
        
        return df
    
    def _create_seasonal_features(self, df):
        """Create seasonal features"""
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df


    
    def _encode_categorical_features(self, df, Training=True):
        """Encode categorical variables. 
        - training=True → compute and save encoders 
        - training=False → load and apply saved encoders
        """
        os.makedirs(self.encoder_dir, exist_ok=True)

        if Training:
            # Service type one-hot
            service_dummies = pd.get_dummies(df['service_type'], prefix='service')
            service_columns = service_dummies.columns.tolist()
            joblib.dump(service_columns, os.path.join(self.encoder_dir, "service_cols.pkl"))
            df = pd.concat([df, service_dummies], axis=1)

            # Borough one-hot
            borough_dummies = pd.get_dummies(df['pickup_borough'], prefix='borough')
            borough_columns = borough_dummies.columns.tolist()
            joblib.dump(borough_columns, os.path.join(self.encoder_dir, "borough_cols.pkl"))
            df = pd.concat([df, borough_dummies], axis=1)

            # Zone frequency encoding
            zone_freq = df['pickup_zone'].value_counts(normalize=True)
            joblib.dump(zone_freq, os.path.join(self.encoder_dir, "zone_freq.pkl"))
            df['zone_freq'] = df['pickup_zone'].map(zone_freq)

        else:
            # Load service_type columns
            service_columns = joblib.load(os.path.join('airflow',self.encoder_dir, "service_cols.pkl"))
            service_dummies = pd.get_dummies(df['service_type'], prefix='service')
            for col in service_columns:
                if col not in service_dummies:
                    service_dummies[col] = 0
            df = pd.concat([df, service_dummies[service_columns]], axis=1)

            # Load borough columns
            borough_columns = joblib.load(os.path.join('airflow', self.encoder_dir, "borough_cols.pkl"))
            borough_dummies = pd.get_dummies(df['pickup_borough'], prefix='borough')
            for col in borough_columns:
                if col not in borough_dummies:
                    borough_dummies[col] = 0
            df = pd.concat([df, borough_dummies[borough_columns]], axis=1)

            # Load zone_freq
            zone_freq = joblib.load(os.path.join('airflow', self.encoder_dir, "zone_freq.pkl"))
            df['zone_freq'] = df['pickup_zone'].map(zone_freq).fillna(0)

        return df

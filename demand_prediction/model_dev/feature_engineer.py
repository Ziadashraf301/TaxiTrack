import os
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from config import DATA_CONFIG, PATH_CONFIG
from logger import setup_logger
from pathlib import Path

log_dir = Path(PATH_CONFIG["logs_dir"])
logger = setup_logger(__name__, f"{log_dir}/pipeline.log")


class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    """Wrap sklearn's LabelEncoder to work inside ColumnTransformer."""
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X, y=None):
        logger.info("Fitting LabelEncoder on pickup_zone with %d unique values", X.nunique())
        self.encoder.fit(X.squeeze())
        return self

    def transform(self, X):
        logger.info("Transforming pickup_zone using fitted LabelEncoder")
        transformed = self.encoder.transform(X.squeeze()).reshape(-1, 1)
        return transformed

    def inverse_transform(self, X):
        return self.encoder.inverse_transform(X.squeeze())


class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.config = DATA_CONFIG
        self.feature_names = []
        self.encoder_dir = PATH_CONFIG["encoder_dir"]
        self.Training = True  # Flag to differentiate between training and inference
        self.preprocessor = None
        self.encoded_feature_names = []
        self._is_fitted = False

    def fit(self, X, y=None):
        """Fit the feature engineer on training data"""
        logger.info("Fitting TimeSeriesFeatureEngineer on %d rows", len(X))
        
        X = X.copy()
        timestamp_col = self.config["timestamp_col"]
        group_cols = self.config["group_cols"]
        
        # Ensure datetime
        X[timestamp_col] = pd.to_datetime(X[timestamp_col])
        
        # Sort by groups and time
        X = X.sort_values(group_cols + [timestamp_col])
        
        # Fit categorical encoder
        categorical_cols = self.config['categorical_features']
        os.makedirs(self.encoder_dir, exist_ok=True)
        
        logger.info("Fitting categorical encoders")
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("service", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ['service_type']),
                ("borough", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ['pickup_borough']),
                ("zone", LabelEncoderWrapper(), ['pickup_zone'])
            ],
            remainder="drop"
        )
        
        self.preprocessor.fit(X[categorical_cols])
        
        # Extract and store encoded feature names
        service_features = self.preprocessor.named_transformers_['service'].get_feature_names_out(['service_type'])
        borough_features = self.preprocessor.named_transformers_['borough'].get_feature_names_out(['pickup_borough'])
        zone_feature = ['zone_encoded']
        self.encoded_feature_names = list(np.concatenate([service_features, borough_features, zone_feature]))
        
        # Save encoder
        joblib.dump(self.preprocessor, os.path.join(self.encoder_dir, "categorical_encoder.pkl"))
        logger.info("Saved categorical encoder to %s", self.encoder_dir)
        
        # Generate feature names by transforming a sample
        sample_transformed = self.transform(X.head(100))
        self.feature_names = [col for col in sample_transformed.columns 
                             if col not in [self.config["timestamp_col"], self.config["target_col"]] + group_cols]
        
        logger.info("Fitted with %d features: %s", len(self.feature_names), self.feature_names[:10])
        self._is_fitted = True
        
        return self

    def transform(self, df):
        """Transform data with feature engineering"""
        logger.info("Starting feature engineering with %d rows", len(df))
        df = df.copy()
        timestamp_col = self.config["timestamp_col"]
        target_col = self.config["target_col"]
        group_cols = self.config["group_cols"]

        # Validate required columns
        required_cols = [timestamp_col] + group_cols + self.config['categorical_features']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Ensure datetime
        logger.info("Converting %s to datetime", timestamp_col)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Sort by groups and time
        logger.info("Sorting by %s + %s", group_cols, timestamp_col)
        df = df.sort_values(group_cols + [timestamp_col]).reset_index(drop=True)

        # Feature engineering steps
        df = self._create_time_features(df, timestamp_col)
        df = self._create_lag_features(df, target_col, group_cols)
        df = self._create_rolling_features(df, target_col, group_cols)
        df = self._create_seasonal_features(df)

        # Encode categoricals
        df = self._encode_categorical_features(df)
        
        # Fill NaN values in features (critical for inference)
        df = self._handle_missing_values(df, target_col, group_cols)

        logger.info("Feature engineering completed. Final shape: %s", df.shape)
        return df

    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        self.Training = True
        self.fit(X, y)
        return self.transform(X)

    def _create_time_features(self, df, timestamp_col):
        """Create time-based features"""
        logger.info("Creating time-based features")
        df['hour'] = df[timestamp_col].dt.hour
        df['dayofweek'] = df[timestamp_col].dt.dayofweek
        df['dayofmonth'] = df[timestamp_col].dt.day
        df['weekofyear'] = df[timestamp_col].dt.isocalendar().week.astype(int)
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
        df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) |
                              ((df['hour'] >= 16) & (df['hour'] <= 19))).astype(int)
        return df

    def _create_lag_features(self, df, target_col, group_cols):
        """Create lag features with forward-fill for missing values"""
        logger.info("Creating lag features: %s", self.config["lag_features"])
        
        # Check if target column exists and has values
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found, skipping lag features")
            return df
        
        for lag in self.config["lag_features"]:
            col_name = f'lag_{lag}h'
            df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
            
            # During inference, fill NaN lags with group mean from available data
            if not self.Training:
                # Fill with the last known value per group (forward fill)
                df[col_name] = df.groupby(group_cols)[col_name].ffill()
                
                # If still NaN (no historical data), use group mean or global mean
                if df[col_name].isna().any():
                    group_means = df.groupby(group_cols)[target_col].transform('mean')
                    df[col_name] = df[col_name].fillna(group_means)
                    df[col_name] = df[col_name].fillna(df[target_col].mean())
                    
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
        """Create seasonal (cyclical) features"""
        logger.info("Creating seasonal (cyclical) features")
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['dayofmonth_sin'] = np.sin(2 * np.pi * df['dayofmonth'] / 30)
        df['dayofmonth_cos'] = np.cos(2 * np.pi * df['dayofmonth'] / 30)
        df['weekofyear_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
        df['weekofyear_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52)
        return df

    def _encode_categorical_features(self, df):
        """Encode categorical variables using saved encoder"""
        logger.info("Encoding categorical features: %s", self.config['categorical_features'])
        categorical_cols = self.config['categorical_features']

        if self.Training and self.preprocessor is None:
            # Should not happen if fit() was called, but handle it
            logger.warning("Training mode but preprocessor not fitted. Fitting now.")
            os.makedirs(self.encoder_dir, exist_ok=True)
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("service", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ['service_type']),
                    ("borough", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ['pickup_borough']),
                    ("zone", LabelEncoderWrapper(), ['pickup_zone'])
                ],
                remainder="drop"
            )
            
            encoded = self.preprocessor.fit_transform(df[categorical_cols])
            
            # Extract feature names
            service_features = self.preprocessor.named_transformers_['service'].get_feature_names_out(['service_type'])
            borough_features = self.preprocessor.named_transformers_['borough'].get_feature_names_out(['pickup_borough'])
            zone_feature = ['zone_encoded']
            self.encoded_feature_names = list(np.concatenate([service_features, borough_features, zone_feature]))
            
            # Save encoder
            joblib.dump(self.preprocessor, os.path.join(self.encoder_dir, "categorical_encoder.pkl"))
            logger.info("Saved categorical encoder to %s", self.encoder_dir)
            
        elif not self.Training:
            # Inference mode: load encoder if not already loaded
            if self.preprocessor is None:
                logger.info("Inference mode: loading saved encoder")
                encoder_path = os.path.join(self.encoder_dir, "categorical_encoder.pkl")
                if not os.path.exists(encoder_path):
                    raise FileNotFoundError(f"Encoder not found at {encoder_path}. Train the model first.")
                self.preprocessor = joblib.load(encoder_path)
                
                # Extract feature names
                service_features = self.preprocessor.named_transformers_['service'].get_feature_names_out(['service_type'])
                borough_features = self.preprocessor.named_transformers_['borough'].get_feature_names_out(['pickup_borough'])
                zone_feature = ['zone_encoded']
                self.encoded_feature_names = list(np.concatenate([service_features, borough_features, zone_feature]))
            
            encoded = self.preprocessor.transform(df[categorical_cols])
        else:
            # Training mode with preprocessor already fitted
            encoded = self.preprocessor.transform(df[categorical_cols])

        # Create encoded dataframe with proper index alignment
        encoded_df = pd.DataFrame(encoded, columns=self.encoded_feature_names, index=df.index)
        
        logger.info("Categorical encoding added %d new features", encoded_df.shape[1])
        
        # Merge back with explicit index reset to avoid issues
        df = df.reset_index(drop=True)
        encoded_df = encoded_df.reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)
        
        return df

    def _handle_missing_values(self, df, target_col, group_cols):
        """Handle missing values in engineered features"""
        logger.info("Handling missing values in features")
        
        # Get all feature columns (exclude target, timestamp, and group columns)
        feature_cols = [col for col in df.columns 
                       if col not in [target_col, self.config["timestamp_col"]] + group_cols + self.config['categorical_features']]
        
        # Fill NaN values
        for col in feature_cols:
            if df[col].isna().any():
                nan_count = df[col].isna().sum()
                logger.debug(f"Filling {nan_count} NaN values in {col}")
                
                # Strategy: forward fill within groups, then backward fill, then use 0
                if 'lag' in col or 'rolling' in col or 'ema' in col:
                    # For temporal features, use group-specific forward/backward fill
                    df[col] = df.groupby(group_cols)[col].ffill()
                    df[col] = df.groupby(group_cols)[col].bfill()
                
                # Final fallback: fill remaining NaNs with 0
                df[col] = df[col].fillna(0)
        
        return df

    def get_feature_names(self):
        """Get list of all feature names after transformation"""
        if not self._is_fitted:
            logger.warning("FeatureEngineer not fitted yet. Feature names may be incomplete.")
        return self.feature_names
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from config import PATH_CONFIG
from logger import setup_logger
from pathlib import Path
import gc

log_dir = Path(PATH_CONFIG["logs_dir"])
logger = setup_logger(__name__, f"{log_dir}/pipeline.log")

class TimeSeriesEncoder:
    def __init__(self):
        self.encoder_dir = PATH_CONFIG["encoder_dir"]
        self.training = True
        self.encoded_feature_names = []
        self.preprocessor = None
        self.scaler = None

    def fit(self, X):
        """Fit encoders + scaler"""
        self.training = True

        logger.info("Fit OneHot Encoders: service_type, time_of_day")
        X_train_encoded = self._encode_categorical_features(X)
        logger.info("Normlization: numerical features")
        self._standardize_features(X_train_encoded)
        
        del X_train_encoded
        gc.collect()

    def transform(self, X):
        """Transform using fitted encoders + scaler"""
        self.training = False
        
        logger.info("Encoding categorical features: service_type, time_of_day")
        X_encoded = self._encode_categorical_features(X)

        logger.info("Standardizing numerical features")
        X_scaled = self._standardize_features(X_encoded)
        return X_scaled

    def fit_transform(self, X):
        """Fit + transform"""
        self.fit(X)
        return self.transform(X)

    def _encode_categorical_features(self, df):
        """Encode service_type, pickup_borough, time_of_day"""

        cat_cols = ['service_type', 'time_of_day']

        if self.training:
            logger.info("Fitting categorical encoder")
            os.makedirs(self.encoder_dir, exist_ok=True)

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("service", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ['service_type']),
                    ("time_of_day", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ['time_of_day'])
                ],
                remainder="drop"
            )

            encoded = self.preprocessor.fit_transform(df[cat_cols])
            joblib.dump(self.preprocessor, os.path.join(self.encoder_dir, "categorical_encoder.pkl"))

        else:
            encoder_path = os.path.join(self.encoder_dir, "categorical_encoder.pkl")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Encoder not found at {encoder_path}")
            self.preprocessor = joblib.load(encoder_path)
            encoded = self.preprocessor.transform(df[cat_cols])

        # Extract encoded feature names
        service_features = self.preprocessor.named_transformers_['service'].get_feature_names_out(['service_type'])
        tod_features = self.preprocessor.named_transformers_['time_of_day'].get_feature_names_out(['time_of_day'])
        self.encoded_feature_names = list(np.concatenate([service_features, tod_features]))

        encoded_df = pd.DataFrame(encoded, columns=self.encoded_feature_names, index=df.index)

        # Drop original categorical cols, keep other features
        exclude_cols = cat_cols
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        df = df[feature_cols].reset_index(drop=True)
        encoded_df = encoded_df.reset_index(drop=True)

        return pd.concat([df, encoded_df], axis=1)

    def _standardize_features(self, df):
        """Standardize all numerical features (after encoding)."""
        if self.training:
            self.scaler = StandardScaler()
            self.scaler.fit(df)
            joblib.dump(self.scaler, os.path.join(self.encoder_dir, "scaler.pkl"))
            return self
        else:
            scaler_path = os.path.join(self.encoder_dir, "scaler.pkl")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            scaled = self.scaler.transform(df)

        return pd.DataFrame(scaled, columns=list(df), index=df.index)

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
from sklearn.preprocessing import OrdinalEncoder

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

        X_train_encoded = self._encode_categorical_features(X)
        # logger.info("Normlization: numerical features")
        # self._standardize_features(X_train_encoded)
        
        del X_train_encoded
        gc.collect()

    def transform(self, X):
        """Transform using fitted encoders + scaler"""
        self.training = False
        
        X_encoded = self._encode_categorical_features(X)

        # logger.info("Standardizing numerical features")
        # X_encoded = self._standardize_features(X_encoded)
        return X_encoded

    def fit_transform(self, X):
        """Fit + transform"""
        self.fit(X)
        return self.transform(X)


    def _encode_categorical_features(self, df):
        """Encode categorical features for LightGBM using native categorical support."""
        
        cat_cols = ['service_type', 'time_of_day', 'pickup_borough', 'pickup_zone']
        
        if self.training:
            logger.info("Preparing categorical features for LightGBM")
            os.makedirs(self.encoder_dir, exist_ok=True)
            
            # Convert to category dtype with explicit categories
            for col in cat_cols:
                # Clean: lowercase, strip whitespace
                df[col] = df[col].astype(str).str.strip().str.lower()
                
                # Handle NaN values
                df[col] = df[col].replace('nan', 'unknown')
                
                # Convert to categorical (learn all categories from training data)
                df[col] = pd.Categorical(df[col])
                
                logger.info(f"{col}: {len(df[col].cat.categories)} unique categories")
            
            # Save the category mappings
            category_mappings = {col: df[col].cat.categories.tolist() for col in cat_cols}
            joblib.dump(category_mappings, os.path.join(self.encoder_dir, "category_mappings.pkl"))
            
            logger.info(f"Saved categorical mappings")
        
        else:
            # Load the category mappings from training
            mappings_path = os.path.join(self.encoder_dir, "category_mappings.pkl")
            if not os.path.exists(mappings_path):
                raise FileNotFoundError(f"Category mappings not found at {mappings_path}")
            
            category_mappings = joblib.load(mappings_path)
            
            # Apply the same categories as training
            for col in cat_cols:
                # Clean: lowercase, strip whitespace
                df[col] = df[col].astype(str).str.strip().str.lower()
                
                # Handle NaN values
                df[col] = df[col].replace('nan', 'unknown')
                
                # Check for unknown categories
                unknown = set(df[col].unique()) - set(category_mappings[col])
                if unknown:
                    logger.warning(f"Unknown categories in {col}: {len(unknown)} categories")
                    logger.debug(f"Examples: {list(unknown)[:5]}")
                    # Map unknown to 'unknown' category
                    df[col] = df[col].apply(lambda x: x if x in category_mappings[col] else 'unknown')
                
                # Set as categorical with the same categories as training
                df[col] = pd.Categorical(df[col], categories=category_mappings[col])
        
        return df


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

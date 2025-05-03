import numpy as np
import pandas as pd
import joblib
import logging

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from catboost import CatBoostRegressor

# Configure logging
logging.basicConfig(filename='demand_model_trainer.log', force=True, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Supported models
MODEL_CLASSES = {
    'bayesian_ridge': BayesianRidge,
    'ridge': Ridge,
    'linear_regression': LinearRegression,
    'random_forest': RandomForestRegressor,
    'decision_tree': DecisionTreeRegressor,
    'xgboost': xgb.XGBRegressor,
    'catboost': CatBoostRegressor
}

def split_data(data, target_column, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def encode_data(X_train, X_test, categorical_cols, method='onehot', y_train=None):
    """
    Encode categorical columns in both training and test datasets.
    
    Parameters:
    - X_train: pd.DataFrame
    - X_test: pd.DataFrame
    - categorical_cols: list of column names to encode
    - method: 'onehot', 'label', or 'target'
    - y_train: Series, required for target encoding
    
    Returns:
    - X_train_encoded: encoded training set
    - X_test_encoded: encoded test set
    """
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    if method == 'onehot':
        transformer = ColumnTransformer(
            transformers=[('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)],
            remainder='passthrough'
        )
        transformer.fit(X_train)
        
        train_encoded = transformer.transform(X_train)
        test_encoded = transformer.transform(X_test)
        
        feature_names = transformer.named_transformers_['ohe'].get_feature_names_out(categorical_cols)
        numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
        all_features = list(feature_names) + numeric_cols

        X_train_encoded = pd.DataFrame(train_encoded, columns=all_features, index=X_train.index)
        X_test_encoded = pd.DataFrame(test_encoded, columns=all_features, index=X_test.index)

    elif method == 'label':
        for col in categorical_cols:
            le = LabelEncoder()
            X_train_encoded[col] = le.fit_transform(X_train[col])
            # Use transform on X_test but handle unseen labels gracefully
            X_test_encoded[col] = X_test[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    elif method == 'target':
        if y_train is None:
            raise ValueError("Target variable is required for target encoding.")
        for col in categorical_cols:
            means = y_train.groupby(X_train[col]).mean()
            X_train_encoded[col] = X_train[col].map(means)
            X_test_encoded[col] = X_test[col].map(means)

    else:
        raise ValueError("Unsupported encoding method.")

    return X_train_encoded, X_test_encoded

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name='ridge', model_params=None):
    """Train and evaluate the model."""
    model_params = model_params or {}

    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Model '{model_name}' is not supported.")

    model = MODEL_CLASSES[model_name](**model_params)
    pipeline = Pipeline([(model_name, model)])

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, scoring='neg_root_mean_squared_error', cv=kf)
    mean_cv_rmse = -np.mean(cv_scores)
    std_cv_rmse = np.std(cv_scores)
    logging.info(f"CV RMSE: {mean_cv_rmse:.2f} ± {std_cv_rmse:.2f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logging.info(f"Model: {model_name}, Params: {model_params}, CV RMSE: {mean_cv_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
    print(f"Model: {model_name}\nCV RMSE: {mean_cv_rmse:.2f}\nTest RMSE: {test_rmse:.2f}")
    return pipeline

def train_on_full_data_and_save(data, target_column, categorical_cols, model_name='ridge', encoding_method='onehot', model_params=None):
    """Train the model on the full dataset and save it."""
    model_params = model_params or {}

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Create preprocessing pipeline
    if encoding_method == 'onehot':
        preprocessor = ColumnTransformer(
            transformers=[('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)],
            remainder='passthrough'
        )
    elif encoding_method == 'label':
        preprocessor = ColumnTransformer(
            transformers=[('label', LabelEncoder(), categorical_cols)],
            remainder='passthrough'
        )
    elif encoding_method == 'target':
        # Target encoding requires y_train, so we need to handle it separately
        def target_encode(X_train, y_train):
            X_train_encoded = X_train.copy()
            for col in categorical_cols:
                means = y_train.groupby(X_train[col]).mean()
                X_train_encoded[col] = X_train[col].map(means)
            return X_train_encoded

        preprocessor = target_encode
    else:
        raise ValueError("Unsupported encoding method.")

    # Create model pipeline
    model = MODEL_CLASSES[model_name](**model_params)

    # If using target encoding, need to pass through preprocessor manually
    if encoding_method != 'target':
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), (model_name, model)])
    else:
        pipeline = Pipeline(steps=[(model_name, model)])

    # Train model
    pipeline.fit(X, y)

    # Save the trained model
    joblib.dump(pipeline, f'{model_name}_pipeline.pkl')
    logging.info(f"✅ Full model trained and saved as '{model_name}_pipeline.pkl'")
    print(f"✅ Full model trained and saved as '{model_name}_pipeline.pkl'")


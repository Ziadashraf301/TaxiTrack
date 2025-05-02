# Import necessary libraries
import numpy as np
import pandas as pd
import joblib
import logging
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import inspect
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb

# Configure logging
logging.basicConfig(filename='model_training.log', force=True , level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Split the data
def split_data(data, target_column, test_size=0.2, random_state=42):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Step 2: Preprocess the data
def preprocess_data(X_train, X_test, categorical_cols, method='onehot', y_train=None):
    if method == 'onehot':
        column_transformer = ColumnTransformer(
            transformers=[
                ('ohe', OneHotEncoder(drop='first'), categorical_cols)
            ],
            remainder='passthrough'
        )
    elif method == 'label':
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
        return X_train, X_test
    elif method == 'target':
        for col in categorical_cols:
            means = y_train.groupby(X_train[col]).mean()
            X_train[col] = X_train[col].map(means)
            X_test[col] = X_test[col].map(means)
        return X_train, X_test
    else:
        raise ValueError("Invalid encoding method")

    X_train_transformed = column_transformer.fit_transform(X_train)
    X_test_transformed = column_transformer.transform(X_test)

    ohe_feature_names = column_transformer.named_transformers_['ohe'].get_feature_names_out(categorical_cols)
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
    all_feature_names = list(ohe_feature_names) + numeric_cols

    X_train_encoded = pd.DataFrame(X_train_transformed, columns=all_feature_names, index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_transformed, columns=all_feature_names, index=X_test.index)

    return X_train_encoded, X_test_encoded


def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name='ridge', model_params=None):
    model_params = model_params or {}

    model_classes = {
        'bayesian_ridge': BayesianRidge,
        'ridge': Ridge,
        'linear_regression': LinearRegression,
        'random_forest': RandomForestRegressor,
        'decision_tree': DecisionTreeRegressor,
        'xgboost': xgb.XGBRegressor,
        'catboost': CatBoostRegressor
    }

    if model_name not in model_classes:
        raise ValueError(f"Model '{model_name}' is not supported.")

    pipeline = Pipeline([
        (model_name, model_classes[model_name](**model_params))
    ])

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_train_scores = cross_val_score(pipeline, X_train, y_train, scoring='neg_root_mean_squared_error', cv=kf)
    mean_cv_train_rmse = -np.mean(cv_train_scores)

    # Fit and evaluate on held-out test set
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    logging.info(f"Model: {model_name} with params: {model_params}")
    logging.info(f"5-Fold CV Mean Train RMSE: {mean_cv_train_rmse:.2f}")
    logging.info(f"Test RMSE: {test_rmse:.2f}")

    print(f"Model: {model_name}")
    print(f"5-Fold CV Mean Train RMSE: {mean_cv_train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")

    return pipeline

# Step 4: Train on full data and save the model
def train_on_full_data_and_save(data, target_column, categorical_cols, model_name='ridge', encoding_method='onehot', model_params=None):
    model_params = model_params or {}

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_encoded, _ = preprocess_data(X, X, categorical_cols, method=encoding_method, y_train=y)

    model_classes = {
        'bayesian_ridge': BayesianRidge,
        'ridge': Ridge,
        'linear_regression': LinearRegression,
        'random_forest': RandomForestRegressor,
        'decision_tree': DecisionTreeRegressor,
        'xgboost': xgb.XGBRegressor,
        'catboost': CatBoostRegressor
    }

    if model_name not in model_classes:
        raise ValueError(f"Model '{model_name}' is not supported.")

    model_class = model_classes[model_name]

    model_instance = model_class(**model_params)

    model = Pipeline([
        (model_name, model_instance)
    ])

    model.fit(X_encoded, y)
    joblib.dump(model, f'{model_name}_pipeline.pkl')

    logging.info(f"✅ Full model trained with {model_params} and saved as '{model_name}_pipeline.pkl'")
    print(f"✅ Full model trained and saved as '{model_name}_pipeline.pkl'")

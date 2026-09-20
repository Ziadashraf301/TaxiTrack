# tests/test_ml_components.py
"""
Unit tests for Machine Learning components:
- TemporalFeatureEngineer (Lookback buffer, lags, rolling aggregates, cyclic calendar features)
- ForecasterFactory, LightGBMForecaster, XGBoostForecaster
- ModelEvaluator (MAE, RMSE, WAPE, R2)
- ONNXModelExporter & ONNX Runtime inference parity validation
"""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from ml.features.temporal import TemporalFeatureEngineer
from ml.models.factory import ForecasterFactory
from ml.models.base import ModelEvaluator
from ml.models.lightgbm import LightGBMForecaster
from ml.models.xgboost import XGBoostForecaster
from ml.serving.onnx import ONNXModelExporter


@pytest.fixture
def synthetic_training_data():
    """Generates synthetic multi-month time series for temporal feature extraction and modeling."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", "2024-03-31 23:00:00", freq="h")
    n = len(dates)

    df = pd.DataFrame({
        "pickup_datetime": dates,
        "pickup_location_id": np.random.choice([10, 20, 30], size=n),
        "pickup_zone": np.random.choice(["Midtown", "JFK", "Harlem"], size=n),
        "pickup_borough": np.random.choice(["Manhattan", "Queens"], size=n),
        "service_type": "yellow",
        "total_trips": np.random.poisson(lam=25, size=n).astype(float),
        "avg_fare_amount": np.random.uniform(15.0, 35.0, size=n),
        "avg_trip_distance": np.random.uniform(2.0, 6.0, size=n),
    })
    return df


def test_temporal_feature_engineer_transformation(synthetic_training_data):
    """Test feature engineering: lookback split, lag generation, and cyclical features."""
    fe = TemporalFeatureEngineer(
        lag_hours=[1, 2, 24],
        rolling_windows=[3, 6],
        group_col="group_id",
        target_col="total_trips",
    )

    X_train, X_val, y_train, y_val = fe.prepare_train_val_split(
        synthetic_training_data, test_months=1
    )

    assert not X_train.empty
    assert not X_val.empty
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)

    # Verify lag and rolling feature columns exist
    for lag in [1, 2, 24]:
        assert f"lag_{lag}h" in X_train.columns
    for w in [3, 6]:
        assert f"rolling_mean_{w}h" in X_train.columns
        assert f"rolling_std_{w}h" in X_train.columns

    # Verify cyclic features
    assert "sin_hour" in X_train.columns
    assert "cos_hour" in X_train.columns
    assert "sin_dow" in X_train.columns
    assert "cos_dow" in X_train.columns


def test_forecaster_factory():
    """Verify ForecasterFactory properly resolves implementations and rejects invalid types."""
    lgb_model = ForecasterFactory.create("lightgbm")
    assert isinstance(lgb_model, LightGBMForecaster)
    assert lgb_model.name == "LightGBM"

    xgb_model = ForecasterFactory.create("xgboost")
    assert isinstance(xgb_model, XGBoostForecaster)
    assert xgb_model.name == "XGBoost"

    with pytest.raises(ValueError, match="Unsupported model architecture"):
        ForecasterFactory.create("random_forest")


def test_lightgbm_train_predict_and_evaluate(synthetic_training_data):
    """Test LightGBM training, non-negative prediction, evaluation, and serialization."""
    fe = TemporalFeatureEngineer(lag_hours=[1, 24], rolling_windows=[3])
    X_train, X_val, y_train, y_val = fe.prepare_train_val_split(
        synthetic_training_data, test_months=1
    )

    model = LightGBMForecaster(params={"n_estimators": 20, "max_depth": 4, "verbose": -1})
    model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=10)

    # Predict
    preds = model.predict(X_val)
    assert len(preds) == len(X_val)
    assert (preds >= 0.0).all(), "Predictions must be non-negative"

    # Evaluate
    metrics = model.evaluate(X_val, y_val)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "wape_pct" in metrics
    assert "r2_score" in metrics
    assert metrics["mae"] >= 0.0

    # Feature importance
    feat_imp = model.get_feature_importances()
    assert not feat_imp.empty
    assert "feature" in feat_imp.columns
    assert "importance" in feat_imp.columns

    # Serialization save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        model.save(model_path)
        assert os.path.exists(model_path)

        reloaded = LightGBMForecaster().load(model_path)
        reloaded_preds = reloaded.predict(X_val)
        np.testing.assert_allclose(preds, reloaded_preds, rtol=1e-5)


def test_onnx_export_and_parity_validation(synthetic_training_data):
    """Test ONNX model export and inference parity between native LightGBM and ONNX Runtime."""
    fe = TemporalFeatureEngineer(lag_hours=[1], rolling_windows=[3])
    X_train, X_val, y_train, y_val = fe.prepare_train_val_split(
        synthetic_training_data, test_months=1
    )

    # Train a fast LightGBM model
    model = LightGBMForecaster(params={"n_estimators": 10, "max_depth": 3, "verbose": -1})
    model.fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = os.path.join(tmpdir, "test_demand.onnx")

        exported_path = ONNXModelExporter.export(
            lgb_model=model.model,
            feature_names=model.feature_names,
            output_path=onnx_file,
        )
        assert os.path.exists(exported_path)
        assert os.path.getsize(exported_path) > 0

        # Validate numerical parity
        sample_X = X_val.head(20)
        is_valid, max_delta = ONNXModelExporter.validate_numerical_parity(
            native_model=model.model,
            onnx_path=exported_path,
            sample_input=sample_X,
            atol=1e-2,
        )
        assert is_valid is True
        assert max_delta < 0.05

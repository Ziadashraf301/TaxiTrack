"""Unit tests for ForecastService."""
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from cachetools import TTLCache
from api.services.forecast_service import ForecastService


@pytest.fixture
def mock_clickhouse():
    ch = MagicMock()
    ch.client = MagicMock()
    return ch


@pytest.fixture
def mock_onnx_session():
    session = MagicMock()
    input_mock = MagicMock()
    input_mock.name = "input"
    session.get_inputs.return_value = [input_mock]
    # Return 24 predictions
    session.run.return_value = [np.array([45.2] * 24)]
    return session


@pytest.fixture
def forecast_service(mock_clickhouse, mock_onnx_session):
    return ForecastService(
        ch=mock_clickhouse,
        cache=TTLCache(maxsize=100, ttl=300),
        onnx_session=mock_onnx_session,
        model_version="test-v1",
    )


def test_predict_success(forecast_service, mock_clickhouse):
    # Generate 192 mock hourly rows for lookback buffer
    dates = pd.date_range("2019-01-01", periods=192, freq="h")
    lookback_df = pd.DataFrame({
        "pickup_date": dates.strftime("%Y-%m-%d"),
        "pickup_hour": dates.hour,
        "pickup_zone": "JFK Airport",
        "pickup_borough": "Queens",
        "service_type": "yellow",
        "total_trips": np.random.uniform(20, 80, size=192),
    })
    mock_clickhouse.client.query_df.return_value = lookback_df

    results = forecast_service.predict(
        pickup_zone="JFK Airport",
        pickup_borough="Queens",
        service_type="yellow",
        horizon_hours=24,
    )

    assert len(results) == 24
    assert all(r.predicted_trips >= 0 for r in results)
    assert all(r.model_version == "test-v1" for r in results)


def test_predict_unknown_zone_raises_key_error(forecast_service, mock_clickhouse):
    # Empty DataFrame simulates unknown or unrecorded zone
    mock_clickhouse.client.query_df.return_value = pd.DataFrame()

    with pytest.raises(KeyError) as exc_info:
        forecast_service.predict(
            pickup_zone="UNKNOWN_ZONE",
            pickup_borough="Nowhere",
            service_type="yellow",
            horizon_hours=24,
        )
    assert "UNKNOWN_ZONE" in str(exc_info.value)


def test_get_historical(forecast_service, mock_clickhouse):
    hist_df = pd.DataFrame([
        {"pickup_date": "2019-01-01", "pickup_hour": 10, "total_trips": 54.0},
        {"pickup_date": "2019-01-01", "pickup_hour": 11, "total_trips": 62.0},
    ])
    mock_clickhouse.client.query_df.return_value = hist_df

    points = forecast_service.get_historical(
        start_date="2019-01-01",
        end_date="2019-01-01",
        pickup_zone="JFK Airport",
        pickup_borough="Queens",
        service_type="yellow",
    )

    assert len(points) == 2
    assert points[0].datetime == "2019-01-01 10:00:00"
    assert points[0].total_trips == 54.0

"""Unit tests for AnalyticsService."""
from unittest.mock import MagicMock
import pandas as pd
import pytest
from cachetools import TTLCache
from api.services.analytics_service import AnalyticsService


@pytest.fixture
def mock_clickhouse():
    ch = MagicMock()
    ch.client = MagicMock()
    return ch


@pytest.fixture
def cache():
    return TTLCache(maxsize=100, ttl=300)


@pytest.fixture
def analytics_service(mock_clickhouse, cache):
    return AnalyticsService(ch=mock_clickhouse, cache=cache)


def test_get_overview_calculates_kpis_and_deltas(analytics_service, mock_clickhouse):
    # Setup mock data for current and prior periods
    curr_df = pd.DataFrame([{
        "total_trips": 1000,
        "total_revenue": 25000.0,
        "total_passengers": 1500,
        "total_tips": 3750.0,
    }])
    prior_df = pd.DataFrame([{
        "total_trips": 800,
        "total_revenue": 20000.0,
        "total_passengers": 1200,
        "total_tips": 3000.0,
    }])
    mock_clickhouse.client.query_df.side_effect = [curr_df, prior_df]

    kpi = analytics_service.get_overview(
        start_date="2019-01-01",
        end_date="2019-01-31",
        borough="Manhattan",
        service_type="yellow",
    )

    assert kpi.total_trips == 1000
    assert kpi.total_revenue == 25000.0
    assert kpi.total_passengers == 1500
    assert kpi.avg_tip_rate == 15.0
    assert kpi.trips_pct_change == 25.0
    assert kpi.revenue_pct_change == 25.0
    assert kpi.passengers_pct_change == 25.0


def test_overview_cache_aside(analytics_service, mock_clickhouse):
    curr_df = pd.DataFrame([{
        "total_trips": 500,
        "total_revenue": 10000.0,
        "total_passengers": 700,
        "total_tips": 1500.0,
    }])
    prior_df = pd.DataFrame([{
        "total_trips": 500,
        "total_revenue": 10000.0,
        "total_passengers": 700,
        "total_tips": 1500.0,
    }])
    mock_clickhouse.client.query_df.side_effect = [curr_df, prior_df]

    # First call - cache miss
    kpi1 = analytics_service.get_overview("2019-02-01", "2019-02-28")
    assert mock_clickhouse.client.query_df.call_count == 2

    # Second call with same arguments - cache hit
    kpi2 = analytics_service.get_overview("2019-02-01", "2019-02-28")
    assert kpi1.total_trips == kpi2.total_trips
    assert mock_clickhouse.client.query_df.call_count == 2  # No extra query!


def test_get_timeseries(analytics_service, mock_clickhouse):
    ts_df = pd.DataFrame([
        {"date": "2019-01-01", "num_trips": 120, "total_revenue": 2400.0, "total_passengers": 180},
        {"date": "2019-01-02", "num_trips": 150, "total_revenue": 3100.0, "total_passengers": 210},
    ])
    mock_clickhouse.client.query_df.return_value = ts_df

    points = analytics_service.get_timeseries("2019-01-01", "2019-01-02", granularity="daily")
    assert len(points) == 2
    assert points[0].date == "2019-01-01"
    assert points[0].num_trips == 120
    assert points[1].total_revenue == 3100.0


def test_get_breakdown(analytics_service, mock_clickhouse):
    bk_df = pd.DataFrame([
        {"borough": "Manhattan", "service_type": "yellow", "num_trips": 800, "total_revenue": 16000.0, "avg_tip_rate": 18.2},
        {"borough": "Queens", "service_type": "green", "num_trips": 200, "total_revenue": 3500.0, "avg_tip_rate": 12.5},
    ])
    mock_clickhouse.client.query_df.return_value = bk_df

    rows = analytics_service.get_breakdown("2019-01-01", "2019-01-31")
    assert len(rows) == 2
    assert rows[0].borough == "Manhattan"
    assert rows[0].avg_tip_rate == 18.2
    assert rows[1].service_type == "green"

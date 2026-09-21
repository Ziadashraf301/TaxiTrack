"""Unit tests for NetworkService."""
from unittest.mock import MagicMock
import pandas as pd
import pytest
from cachetools import TTLCache
from api.services.network_service import NetworkService


@pytest.fixture
def mock_clickhouse():
    ch = MagicMock()
    ch.client = MagicMock()
    return ch


@pytest.fixture
def network_service(mock_clickhouse):
    return NetworkService(
        ch=mock_clickhouse,
        cache=TTLCache(maxsize=100, ttl=3600),
    )


def test_get_top_corridors(network_service, mock_clickhouse):
    df_corridors = pd.DataFrame([
        {
            "source_location": "JFK Airport",
            "target_location": "Midtown Center",
            "trip_count": 1500,
            "avg_distance": 17.5,
            "avg_duration_minutes": 42.0,
        },
        {
            "source_location": "LaGuardia Airport",
            "target_location": "Upper East Side",
            "trip_count": 1200,
            "avg_distance": 8.2,
            "avg_duration_minutes": 25.0,
        },
    ])
    mock_clickhouse.client.query_df.return_value = df_corridors

    rows = network_service.get_top_corridors("2019-01", top_n=2)
    assert len(rows) == 2
    assert rows[0].source == "JFK Airport"
    assert rows[0].target == "Midtown Center"
    assert rows[0].trip_count == 1500


def test_get_centrality_computes_network_metrics(network_service, mock_clickhouse):
    # Construct a connected triangular network
    df_graph = pd.DataFrame([
        {"source_location": "ZoneA", "target_location": "ZoneB", "trip_count": 50, "avg_distance": 2.0, "sum_total_amounts": 500.0, "avg_duration_minutes": 10.0},
        {"source_location": "ZoneB", "target_location": "ZoneC", "trip_count": 30, "avg_distance": 3.0, "sum_total_amounts": 300.0, "avg_duration_minutes": 15.0},
        {"source_location": "ZoneC", "target_location": "ZoneA", "trip_count": 40, "avg_distance": 2.5, "sum_total_amounts": 400.0, "avg_duration_minutes": 12.0},
    ])
    mock_clickhouse.client.query_df.return_value = df_graph

    centrality_rows = network_service.get_centrality("201901")
    assert len(centrality_rows) == 3
    zones = {r.zone for r in centrality_rows}
    assert zones == {"ZoneA", "ZoneB", "ZoneC"}
    assert all(r.pagerank > 0 for r in centrality_rows)

"""Network service for spatial corridor analytics and NetworkX graph metrics."""
from typing import Any, List
import pandas as pd
from api.schemas.network import CentralityRow, CorridorRow
from api.services.base import BaseDataService
from core.logging import get_logger
from ml.graph.spatial import SpatialNetworkAnalyzer

logger = get_logger(__name__)


class NetworkService(BaseDataService):
    """Provides spatial graph metrics and origin-destination corridor analysis."""

    def __init__(self, ch: Any = None, cache: Any = None):
        super().__init__(ch=ch, cache=cache)

    def _clean_month(self, pickup_month: str) -> str:
        """Standardize month representations (e.g. '2019-01' -> '201901')."""
        return str(pickup_month).replace("-", "").strip()

    def get_top_corridors(self, pickup_month: str, top_n: int = 20) -> List[CorridorRow]:
        """Fetch top origin-destination transit corridors for a given month."""
        month_clean = self._clean_month(pickup_month)
        cache_key = f"corridors:{month_clean}:{top_n}"

        def _fetch() -> List[CorridorRow]:
            query = f"""
                SELECT
                    source_location,
                    target_location,
                    trip_count,
                    avg_distance,
                    avg_duration_minutes
                FROM data_warehouse.trip_location_network_metrics
                WHERE toString(pickup_month) = '{month_clean}'
                ORDER BY trip_count DESC
                LIMIT {top_n}
            """
            df = self.ch.client.query_df(query)
            if df is None or df.empty:
                return []

            rows = []
            for _, r in df.iterrows():
                rows.append(
                    CorridorRow(
                        source=str(r["source_location"]),
                        target=str(r["target_location"]),
                        trip_count=int(r["trip_count"]),
                        avg_distance=float(r["avg_distance"]),
                        avg_duration=float(r["avg_duration_minutes"]),
                    )
                )
            return rows

        return self._cached(cache_key, _fetch)

    def get_centrality(self, pickup_month: str) -> List[CentralityRow]:
        """Compute zone-level NetworkX centrality metrics (PageRank, flow degree, betweenness)."""
        month_clean = self._clean_month(pickup_month)
        cache_key = f"centrality:{month_clean}"

        def _fetch() -> List[CentralityRow]:
            query = f"""
                SELECT
                    source_location,
                    target_location,
                    trip_count,
                    avg_distance,
                    sum_total_amounts,
                    avg_duration_minutes
                FROM data_warehouse.trip_location_network_metrics
                WHERE toString(pickup_month) = '{month_clean}'
            """
            df = self.ch.client.query_df(query)
            if df is None or df.empty:
                return []

            analyzer = SpatialNetworkAnalyzer()
            analyzer.build_graph(df)
            analyzer.compute_metrics()

            if analyzer.summary_df is None or analyzer.summary_df.empty:
                return []

            results = []
            for _, r in analyzer.summary_df.iterrows():
                results.append(
                    CentralityRow(
                        zone=str(r["zone"]),
                        pagerank=float(r["pagerank"]),
                        in_degree=float(r["in_degree_centrality"]),
                        out_degree=float(r["out_degree_centrality"]),
                        asymmetry_score=float(r["asymmetry_score"]),
                        betweenness=float(r.get("betweenness_centrality", 0.0)),
                    )
                )
            return results

        return self._cached(cache_key, _fetch)

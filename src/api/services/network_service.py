"""Network service for spatial corridor analytics and NetworkX graph metrics."""
from typing import Any, List, Optional
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

    def get_top_corridors(
        self,
        pickup_month: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        zone: Optional[str] = None,
        top_n: int = 20,
    ) -> List[CorridorRow]:
        """Fetch top origin-destination transit corridors across the specified date range and zone."""
        start_m = None
        end_m = None
        if start_date:
            start_m = str(start_date).replace("-", "")[:6]
        if end_date:
            end_m = str(end_date).replace("-", "")[:6]
        if pickup_month and not start_m:
            start_m = end_m = self._clean_month(pickup_month)
        if not start_m:
            start_m = "201901"
        if not end_m:
            end_m = start_m

        zone_clean = zone.strip() if zone and zone.lower() not in ("all", "any") else None
        cache_key = f"corridors:{start_m}:{end_m}:{zone_clean}:{top_n}"

        def _fetch() -> List[CorridorRow]:
            where_clauses = [
                f"toString(pickup_month) >= '{start_m}'",
                f"toString(pickup_month) <= '{end_m}'",
            ]
            if zone_clean:
                where_clauses.append(f"(lower(trim(source_location)) = '{zone_clean.lower()}' OR lower(trim(target_location)) = '{zone_clean.lower()}')")

            where_sql = " AND ".join(where_clauses)
            query = f"""
                SELECT
                    source_location,
                    target_location,
                    coalesce(sum(trip_count), 0) AS trip_count,
                    round(avg(avg_distance), 2) AS avg_distance,
                    round(avg(avg_duration_minutes), 2) AS avg_duration_minutes
                FROM data_warehouse.trip_location_network_metrics
                WHERE {where_sql}
                GROUP BY source_location, target_location
                ORDER BY trip_count DESC
                LIMIT {top_n}
            """
            df = self.ch.client.query_df(query)
            if (df is None or df.empty) and zone_clean:
                fallback_where = f"toString(pickup_month) >= '{start_m}' AND toString(pickup_month) <= '{end_m}'"
                query_fallback = f"""
                    SELECT
                        source_location,
                        target_location,
                        coalesce(sum(trip_count), 0) AS trip_count,
                        round(avg(avg_distance), 2) AS avg_distance,
                        round(avg(avg_duration_minutes), 2) AS avg_duration_minutes
                    FROM data_warehouse.trip_location_network_metrics
                    WHERE {fallback_where}
                    GROUP BY source_location, target_location
                    ORDER BY trip_count DESC
                    LIMIT {top_n}
                """
                df = self.ch.client.query_df(query_fallback)

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

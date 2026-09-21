"""Analytics service querying ClickHouse mart_daily_taxi_performance."""
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
from api.schemas.analytics import BreakdownRow, KPIResponse, TimeseriesPoint
from api.services.base import BaseDataService
from core.logging import get_logger

logger = get_logger(__name__)


class AnalyticsService(BaseDataService):
    """High-performance analytics data provider for NYC taxi operational metrics."""

    def _build_where_clause(
        self,
        start_date: str,
        end_date: str,
        borough: Optional[str] = None,
        service_type: Optional[str] = None,
    ) -> str:
        clauses = [
            f"pickup_date >= '{start_date}'",
            f"pickup_date <= '{end_date}'",
        ]
        if borough and borough.lower() not in ("all", "any"):
            clauses.append(f"lower(borough) = '{borough.lower().strip()}'")
        if service_type and service_type.lower() not in ("all", "any"):
            clauses.append(f"lower(service_type) = '{service_type.lower().strip()}'")
        return " AND ".join(clauses)

    def get_overview(
        self,
        start_date: str,
        end_date: str,
        borough: Optional[str] = None,
        service_type: Optional[str] = None,
    ) -> KPIResponse:
        """Calculate high-level KPIs and period-over-period percentage deltas."""
        cache_key = f"overview:{start_date}:{end_date}:{borough}:{service_type}"

        def _fetch() -> KPIResponse:
            where_sql = self._build_where_clause(start_date, end_date, borough, service_type)
            query_curr = f"""
                SELECT
                    coalesce(sum(num_trips), 0) AS total_trips,
                    coalesce(sum(total_revenue), 0.0) AS total_revenue,
                    coalesce(sum(total_passengers), 0) AS total_passengers,
                    coalesce(sum(total_tips), 0.0) AS total_tips
                FROM data_warehouse.mart_daily_taxi_performance
                WHERE {where_sql}
            """
            curr_df = self.ch.client.query_df(query_curr)
            curr_trips = int(curr_df["total_trips"].iloc[0]) if not curr_df.empty else 0
            curr_rev = float(curr_df["total_revenue"].iloc[0]) if not curr_df.empty else 0.0
            curr_pass = int(curr_df["total_passengers"].iloc[0]) if not curr_df.empty else 0
            curr_tips = float(curr_df["total_tips"].iloc[0]) if not curr_df.empty else 0.0

            curr_tip_rate = round((curr_tips / curr_rev * 100), 2) if curr_rev > 0 else 0.0

            # Calculate prior period
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                duration = (end_dt - start_dt).days + 1
                prior_end_dt = start_dt - timedelta(days=1)
                prior_start_dt = prior_end_dt - timedelta(days=duration - 1)

                prior_where = self._build_where_clause(
                    prior_start_dt.strftime("%Y-%m-%d"),
                    prior_end_dt.strftime("%Y-%m-%d"),
                    borough,
                    service_type,
                )
                query_prior = f"""
                    SELECT
                        coalesce(sum(num_trips), 0) AS total_trips,
                        coalesce(sum(total_revenue), 0.0) AS total_revenue,
                        coalesce(sum(total_passengers), 0) AS total_passengers,
                        coalesce(sum(total_tips), 0.0) AS total_tips
                    FROM data_warehouse.mart_daily_taxi_performance
                    WHERE {prior_where}
                """
                prior_df = self.ch.client.query_df(query_prior)
                prior_trips = int(prior_df["total_trips"].iloc[0]) if not prior_df.empty else 0
                prior_rev = float(prior_df["total_revenue"].iloc[0]) if not prior_df.empty else 0.0
                prior_pass = int(prior_df["total_passengers"].iloc[0]) if not prior_df.empty else 0
                prior_tips = float(prior_df["total_tips"].iloc[0]) if not prior_df.empty else 0.0
                prior_tip_rate = round((prior_tips / prior_rev * 100), 2) if prior_rev > 0 else 0.0
            except Exception as e:
                logger.warning(f"Could not compute prior period KPIs: {e}")
                prior_trips = prior_rev = prior_pass = prior_tip_rate = None

            def calc_pct_change(curr: float, prior: Optional[float]) -> Optional[float]:
                if prior is None or prior == 0:
                    return None
                return round(((curr - prior) / prior) * 100, 2)

            return KPIResponse(
                total_trips=curr_trips,
                total_revenue=round(curr_rev, 2),
                total_passengers=curr_pass,
                avg_tip_rate=curr_tip_rate,
                trips_pct_change=calc_pct_change(curr_trips, prior_trips),
                revenue_pct_change=calc_pct_change(curr_rev, prior_rev),
                passengers_pct_change=calc_pct_change(curr_pass, prior_pass),
                tip_rate_pct_change=calc_pct_change(curr_tip_rate, prior_tip_rate),
            )

        return self._cached(cache_key, _fetch)

    def get_timeseries(
        self,
        start_date: str,
        end_date: str,
        granularity: str = "daily",
        borough: Optional[str] = None,
        service_type: Optional[str] = None,
    ) -> List[TimeseriesPoint]:
        """Fetch aggregated trip volume and revenue timeseries by day or month."""
        cache_key = f"timeseries:{start_date}:{end_date}:{granularity}:{borough}:{service_type}"

        def _fetch() -> List[TimeseriesPoint]:
            where_sql = self._build_where_clause(start_date, end_date, borough, service_type)
            if granularity.lower() == "monthly":
                date_expr = "formatDateTime(pickup_date, '%Y-%m')"
            else:
                date_expr = "toString(pickup_date)"

            query = f"""
                SELECT
                    {date_expr} AS date,
                    coalesce(sum(num_trips), 0) AS num_trips,
                    round(coalesce(sum(total_revenue), 0.0), 2) AS total_revenue,
                    coalesce(sum(total_passengers), 0) AS total_passengers
                FROM data_warehouse.mart_daily_taxi_performance
                WHERE {where_sql}
                GROUP BY date
                ORDER BY date ASC
            """
            df = self.ch.client.query_df(query)
            if df.empty:
                return []

            points = []
            for _, row in df.iterrows():
                points.append(
                    TimeseriesPoint(
                        date=str(row["date"]),
                        num_trips=int(row["num_trips"]),
                        total_revenue=float(row["total_revenue"]),
                        total_passengers=int(row["total_passengers"]),
                    )
                )
            return points

        return self._cached(cache_key, _fetch)

    def get_breakdown(
        self,
        start_date: str,
        end_date: str,
    ) -> List[BreakdownRow]:
        """Fetch borough and service type cross-tabulation breakdown."""
        cache_key = f"breakdown:{start_date}:{end_date}"

        def _fetch() -> List[BreakdownRow]:
            where_sql = self._build_where_clause(start_date, end_date)
            query = f"""
                SELECT
                    borough,
                    service_type,
                    coalesce(sum(num_trips), 0) AS num_trips,
                    round(coalesce(sum(total_revenue), 0.0), 2) AS total_revenue,
                    round(if(sum(total_revenue) > 0, (sum(total_tips) / sum(total_revenue)) * 100.0, 0.0), 2) AS avg_tip_rate
                FROM data_warehouse.mart_daily_taxi_performance
                WHERE {where_sql} AND borough != ''
                GROUP BY borough, service_type
                ORDER BY num_trips DESC
            """
            df = self.ch.client.query_df(query)
            if df.empty:
                return []

            rows = []
            for _, row in df.iterrows():
                rows.append(
                    BreakdownRow(
                        borough=str(row["borough"]),
                        service_type=str(row["service_type"]),
                        num_trips=int(row["num_trips"]),
                        total_revenue=float(row["total_revenue"]),
                        avg_tip_rate=float(row["avg_tip_rate"]),
                    )
                )
            return rows

        return self._cached(cache_key, _fetch)

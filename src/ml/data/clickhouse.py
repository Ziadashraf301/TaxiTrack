# src/ml/data/clickhouse.py
"""
ClickHouse Feature & Network Repository (Repository Pattern)
Concrete implementation of BaseDemandRepository and BaseNetworkRepository for ClickHouse.
"""
import time
from typing import Optional, Tuple
import pandas as pd
from core.clickhouse import ClickHouseService
from core.logging import get_logger
from ml.data.base import BaseDemandRepository, BaseNetworkRepository

logger = get_logger(__name__)


class ClickHouseFeatureRepository(BaseDemandRepository, BaseNetworkRepository):
    """Repository for querying ClickHouse feature marts for ML forecasting and spatial graphs."""

    def __init__(self, clickhouse_service: Optional[ClickHouseService] = None):
        self.clickhouse = clickhouse_service or ClickHouseService()

    def get_demand_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        service_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch aggregated demand data from mart_demand_prediction.
        """
        where_clauses = ["pickup_zone != ''", "pickup_borough != ''"]
        if start_date:
            where_clauses.append(f"pickup_date >= '{start_date}'")
        if end_date:
            where_clauses.append(f"pickup_date < '{end_date}'")
        if service_type:
            where_clauses.append(f"service_type = '{service_type}'")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query = f"""
            SELECT
                pickup_date,
                pickup_hour,
                pickup_zone,
                pickup_borough,
                service_type,
                total_trips
            FROM data_warehouse.mart_demand_prediction
            {where_sql}
            ORDER BY pickup_date, pickup_hour, pickup_zone
        """

        logger.info(f"Querying demand features from mart_demand_prediction (start={start_date}, end={end_date})...")
        start_time = time.perf_counter()

        result = self.clickhouse.client.query_df(query)
        duration = time.perf_counter() - start_time

        if result.empty:
            logger.warning("No records found in mart_demand_prediction matching criteria.")
            return pd.DataFrame()

        # Synthesize hourly pickup_datetime for temporal feature engineering
        result["pickup_datetime"] = pd.to_datetime(result["pickup_date"]) + pd.to_timedelta(
            result["pickup_hour"], unit="h"
        )

        min_dt = result["pickup_datetime"].min()
        max_dt = result["pickup_datetime"].max()
        unique_zones = result["pickup_zone"].nunique()

        logger.info(
            f"Retrieved {len(result):,} demand records in {duration:.3f}s. "
            f"Date Range: {min_dt} to {max_dt} | Unique Zones: {unique_zones}"
        )
        return result

    def get_network_metrics(
        self,
        pickup_month: Optional[str] = None,
        min_trips: int = 10,
    ) -> pd.DataFrame:
        """
        Fetch origin-destination route volume metrics for graph analysis.
        """
        where_clauses = [
            f"trip_count >= {min_trips}",
            "source_location != ''",
            "target_location != ''",
        ]
        if pickup_month:
            where_clauses.append(f"pickup_month = '{pickup_month}'")

        where_sql = f"WHERE {' AND '.join(where_clauses)}"

        query = f"""
            SELECT
                source_location,
                target_location,
                sum(trip_count) AS trip_count,
                round(avg(avg_distance), 2) AS avg_distance,
                round(sum(sum_total_amounts), 2) AS sum_total_amounts,
                round(avg(avg_duration_minutes), 2) AS avg_duration_minutes
            FROM data_warehouse.trip_location_network_metrics
            {where_sql}
            GROUP BY source_location, target_location
            ORDER BY trip_count DESC
        """

        logger.info(f"Querying network location metrics (month={pickup_month}, min_trips={min_trips})...")
        start_time = time.perf_counter()
        result = self.clickhouse.client.query_df(query)
        duration = time.perf_counter() - start_time

        logger.info(f"Retrieved {len(result):,} route corridors in {duration:.3f}s.")
        return result

    def get_demand_summary(self) -> Tuple[Optional[str], Optional[str], int]:
        """Return (min_date, max_date, total_records) from mart_demand_prediction."""
        try:
            res = self.clickhouse.client.query(
                "SELECT min(pickup_date), max(pickup_date), count(*) FROM data_warehouse.mart_demand_prediction"
            ).result_rows
            if res and res[0][2] > 0:
                return str(res[0][0]), str(res[0][1]), int(res[0][2])
            return None, None, 0
        except Exception as e:
            logger.warning(f"Could not fetch demand summary: {e}")
            return None, None, 0

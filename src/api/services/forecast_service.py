"""Forecast service providing historical demand retrieval and ONNX runtime inference."""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from api.schemas.forecast import ForecastPoint, HistoricalPoint
from api.services.base import BaseDataService
from core.logging import get_logger
from ml.features.temporal import TemporalFeatureEngineer

logger = get_logger(__name__)


class ForecastService(BaseDataService):
    """Orchestrates demand prediction queries and ONNX model inferences."""

    def __init__(
        self,
        ch: Any = None,
        cache: Any = None,
        onnx_session: Any = None,
        feature_engineer: Optional[TemporalFeatureEngineer] = None,
        model_version: str = "v1",
    ):
        super().__init__(ch=ch, cache=cache)
        self.onnx_session = onnx_session
        self.feature_engineer = feature_engineer
        self.model_version = model_version

    def get_all_zones(self) -> List[Dict[str, str]]:
        """Retrieve list of all distinct boroughs and zones present in the data warehouse."""
        cache_key = "forecast:zones:all"

        def _fetch() -> List[Dict[str, str]]:
            query = """
                SELECT DISTINCT
                    pickup_borough AS borough,
                    pickup_zone AS zone
                FROM data_warehouse.mart_demand_prediction
                WHERE pickup_zone != '' AND pickup_borough != ''
                ORDER BY pickup_borough ASC, pickup_zone ASC
            """
            df = self.ch.client.query_df(query)
            if df.empty:
                return []
            return [{"borough": str(r["borough"]), "zone": str(r["zone"])} for _, r in df.iterrows()]

        return self._cached(cache_key, _fetch)

    def get_historical(
        self,
        start_date: str,
        end_date: str,
        pickup_zone: str,
        pickup_borough: str,
        service_type: str,
    ) -> List[HistoricalPoint]:
        """Query historical observed demand for a specific zone from mart_demand_prediction."""
        cache_key = f"hist:{start_date}:{end_date}:{pickup_zone}:{pickup_borough}:{service_type}".lower()

        def _fetch() -> List[HistoricalPoint]:
            zone = pickup_zone.lower().strip()
            borough = pickup_borough.lower().strip()
            service = service_type.lower().strip()
            service_clean = "yellow_trip" if service == "yellow" else ("green_trip" if service == "green" else service)

            query = f"""
                SELECT
                    pickup_date,
                    pickup_hour,
                    total_trips
                FROM data_warehouse.mart_demand_prediction
                WHERE lower(trim(pickup_zone)) = '{zone}'
                  AND lower(trim(pickup_borough)) = '{borough}'
                  AND (lower(trim(service_type)) = '{service_clean}' OR lower(trim(service_type)) = '{service}')
                  AND pickup_date >= '{start_date}'
                  AND pickup_date <= '{end_date}'
                ORDER BY pickup_date ASC, pickup_hour ASC
            """
            df = self.ch.client.query_df(query)
            if df.empty:
                return []

            points = []
            for _, row in df.iterrows():
                dt_str = f"{row['pickup_date']} {int(row['pickup_hour']):02d}:00:00"
                points.append(
                    HistoricalPoint(
                        datetime=dt_str,
                        total_trips=float(row["total_trips"]),
                    )
                )
            return points

        return self._cached(cache_key, _fetch)

    def predict(
        self,
        pickup_zone: str,
        pickup_borough: str,
        service_type: str,
        horizon_hours: int = 720,
        end_date: Optional[str] = None,
    ) -> List[ForecastPoint]:
        """
        Generate forward-looking demand predictions using the production ONNX model.
        Constructs group_id, retrieves lookback buffer up to end_date, transforms features, and runs inference.
        """
        zone = pickup_zone.strip()
        borough = pickup_borough.strip()
        service = service_type.strip()
        service_clean = "yellow_trip" if service.lower() == "yellow" else ("green_trip" if service.lower() == "green" else service.lower())
        group_id = f"{zone}__{borough}__{service_clean}".lower()

        logger.info(f"Predicting demand for group '{group_id}' (horizon={horizon_hours}h, end_date={end_date})...")

        # 1. Fetch lookback buffer (192h > 168h max lag)
        # Filter strictly on or before end_date to anchor forecast immediately following the chosen time window
        date_filter = f"AND pickup_date <= '{end_date}'" if end_date else "AND pickup_date <= today()"
        query = f"""
            SELECT
                pickup_date,
                pickup_hour,
                pickup_zone,
                pickup_borough,
                service_type,
                total_trips
            FROM data_warehouse.mart_demand_prediction
            WHERE lower(trim(pickup_zone)) = '{zone.lower()}'
              AND lower(trim(pickup_borough)) = '{borough.lower()}'
              AND (lower(trim(service_type)) = '{service_clean}' OR lower(trim(service_type)) = '{service.lower()}')
              {date_filter}
            ORDER BY pickup_date DESC, pickup_hour DESC
            LIMIT 192
        """
        df_lookback = self.ch.client.query_df(query)

        if df_lookback is None or df_lookback.empty:
            raise KeyError(
                f"No historical demand records found for zone='{pickup_zone}', "
                f"borough='{pickup_borough}', service='{service_type}'" + (f" on or before {end_date}" if end_date else "")
            )

        # Sort chronologically ascending
        df_lookback = df_lookback.sort_values(
            by=["pickup_date", "pickup_hour"]
        ).reset_index(drop=True)

        df_lookback["pickup_datetime"] = (
            pd.to_datetime(df_lookback["pickup_date"])
            + pd.to_timedelta(df_lookback["pickup_hour"].astype(int), unit="h")
        )

        last_dt = df_lookback["pickup_datetime"].max()
        if end_date:
            try:
                target_end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=0, second=0)
                if last_dt < target_end_dt:
                    last_dt = target_end_dt
            except Exception:
                pass

        future_dts = [last_dt + timedelta(hours=i) for i in range(1, horizon_hours + 1)]

        # 2. Build future placeholder rows
        future_df = pd.DataFrame({
            "pickup_datetime": future_dts,
            "pickup_date": [dt.strftime("%Y-%m-%d") for dt in future_dts],
            "pickup_hour": [dt.hour for dt in future_dts],
            "pickup_zone": zone,
            "pickup_borough": borough,
            "service_type": service_clean,
            "total_trips": [df_lookback["total_trips"].tail(24).mean()] * horizon_hours,
        })

        # Combine lookback buffer + future rows for rolling/lag computation
        combined = pd.concat([df_lookback, future_df], ignore_index=True)

        fe = self.feature_engineer or TemporalFeatureEngineer()
        if not getattr(fe, "_is_fitted", False):
            # If not yet fitted in test environment, fit on lookback
            fe.fit(df_lookback)

        X, _, timestamps = fe.prepare_matrices(combined)

        # Filter strictly for the future horizon
        future_mask = pd.to_datetime(timestamps).isin(future_dts)
        X_future = X[future_mask].copy().reset_index(drop=True)
        times_future = timestamps[future_mask].tolist()

        # 3. ONNX inference
        if self.onnx_session is not None:
            # Handle categorical columns by encoding to codes
            numeric_df = X_future.copy()
            for col in numeric_df.select_dtypes(include=["category"]).columns:
                numeric_df[col] = numeric_df[col].cat.codes.astype(np.float32)
            input_array = numeric_df.values.astype(np.float32)

            input_name = self.onnx_session.get_inputs()[0].name
            raw_preds = self.onnx_session.run(None, {input_name: input_array})[0].flatten()
            preds = np.clip(raw_preds, 0.0, None)
        else:
            # Fallback for mocking/testing without ONNX model file
            logger.warning("ONNX session not initialized; returning baseline predictions.")
            preds = np.array([round(df_lookback["total_trips"].mean(), 1)] * len(X_future))

        results = []
        for dt_val, pred_val in zip(times_future, preds):
            dt_str = pd.to_datetime(dt_val).strftime("%Y-%m-%d %H:%M:%S")
            results.append(
                ForecastPoint(
                    datetime=dt_str,
                    predicted_trips=round(float(pred_val), 1),
                    model_version=self.model_version,
                )
            )

        return results

# src/ml/features/temporal.py
"""
Temporal Feature Engineer (Feature Extraction & Encoding)
Generates calendar features, cyclical encodings, lag metrics, and rolling windows.
Adheres to BaseFeatureEngineer contract with zero temporal leakage via Lookback Buffer architecture.
"""
import os
import joblib
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from ml.features.base import BaseFeatureEngineer
from core.logging import get_logger

logger = get_logger(__name__)


class TemporalFeatureEngineer(BaseFeatureEngineer):
    """
    Transforms raw time-series records from mart_demand_prediction into a model-ready
    feature matrix with cyclical encodings, lag offsets, and rolling statistical windows.
    Eliminates target-encoding leakage and avoids redundant double-splitting via Lookback Buffer.
    """

    def __init__(
        self,
        lag_hours: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        group_col: str = "group_id",
        target_col: str = "total_trips",
        timestamp_col: str = "pickup_datetime",
    ):
        self.lag_hours = lag_hours or [24, 168]
        self.rolling_windows = rolling_windows or [6, 24]
        self.group_col = group_col
        self.target_col = target_col
        self.timestamp_col = timestamp_col

        self.category_mappings: Dict[str, List[str]] = {}
        self.feature_columns: List[str] = []
        self._is_fitted: bool = False

    def _create_group_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite spatial-fleet group key: zone__borough__fleet."""
        df = df.copy()
        if self.group_col not in df.columns:
            if all(c in df.columns for c in ["pickup_zone", "pickup_borough", "service_type"]):
                df[self.group_col] = (
                    df["pickup_zone"].astype(str)
                    + "__"
                    + df["pickup_borough"].astype(str)
                    + "__"
                    + df["service_type"].astype(str)
                ).str.lower().str.strip()
            elif "pickup_location_id" in df.columns:
                df[self.group_col] = df["pickup_location_id"].astype(str)
            else:
                df[self.group_col] = "default_group"
        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create calendar and cyclical sin/cos features from pickup_datetime."""
        dt = pd.to_datetime(df[self.timestamp_col])
        hour = dt.dt.hour
        dow = dt.dt.dayofweek

        df["hour"] = hour
        df["dayofweek"] = dow
        df["dayofmonth"] = dt.dt.day
        df["month"] = dt.dt.month
        df["is_weekend"] = dow.isin([5, 6]).astype(int)
        df["is_rush_hour"] = (
            ((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))
        ).astype(int)

        # Cyclical sin/cos encodings (smooth continuity across 23h -> 0h and Sun -> Mon)
        df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
        df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
        df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
        df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)

        # Time-of-day category
        def categorize_time_of_day(h: int) -> str:
            if 6 <= h < 10:
                return "morning"
            elif 10 <= h < 16:
                return "midday"
            elif 16 <= h < 21:
                return "evening"
            return "night"

        df["time_of_day"] = hour.apply(categorize_time_of_day)
        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lag demand features per spatial-fleet group with zero future leakage.
        Missing lag values at the start of a series are left as NaN for native GBDT handling.
        """
        df = df.sort_values([self.group_col, self.timestamp_col]).reset_index(drop=True)

        for lag in self.lag_hours:
            col_name = f"lag_{lag}h"
            df[col_name] = df.groupby(self.group_col)[self.target_col].shift(lag).astype(np.float32)
        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling mean and standard deviation strictly per spatial-fleet group.
        Guarantees zero lookahead leakage by shifting the rolling window by min(lag_hours)
        (e.g. 24h), ensuring features are strictly aligned with the day-ahead forecast horizon.
        """
        shift_horizon = min(self.lag_hours) if self.lag_hours else 24

        for window in self.rolling_windows:
            mean_col = f"rolling_mean_{window}h"
            std_col = f"rolling_std_{window}h"

            df[mean_col] = (
                df.groupby(self.group_col)[self.target_col]
                .transform(lambda s: s.shift(shift_horizon).rolling(window=window, min_periods=1).mean())
                .astype(np.float32)
            )
            df[std_col] = (
                df.groupby(self.group_col)[self.target_col]
                .transform(lambda s: s.shift(shift_horizon).rolling(window=window, min_periods=1).std())
                .astype(np.float32)
            )
        return df

    def fit(self, df: pd.DataFrame) -> "TemporalFeatureEngineer":
        """Learn categorical mappings strictly from training set."""
        logger.info(f"Fitting TemporalFeatureEngineer on {len(df):,} rows...")
        processed = self._create_group_id(df)
        processed = self._create_time_features(processed)

        # Learn category mappings
        cat_cols = ["time_of_day", self.group_col]
        for col in cat_cols:
            cleaned = processed[col].astype(str).str.strip().str.lower().replace("nan", "unknown")
            categories = sorted(cleaned.unique().tolist())
            if "unknown" not in categories:
                categories.append("unknown")
            self.category_mappings[col] = categories

        self._is_fitted = True
        logger.info(
            f"Fitted categorical mappings: {len(self.category_mappings.get(self.group_col, []))} unique groups, "
            f"{len(self.category_mappings.get('time_of_day', []))} time bins."
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw input DataFrame into fully engineered feature matrix."""
        if not self._is_fitted:
            raise RuntimeError("TemporalFeatureEngineer must be fitted before calling transform()!")

        df = self._create_group_id(df)
        df = self._create_time_features(df)

        if self.target_col in df.columns:
            df = self._create_lag_features(df)
            df = self._create_rolling_features(df)

        # Encode categoricals using fitted mapping categories
        for col, categories in self.category_mappings.items():
            if col in df.columns:
                cleaned = df[col].astype(str).str.strip().str.lower().replace("nan", "unknown")
                cleaned = cleaned.apply(lambda x: x if x in categories else "unknown")
                df[col] = pd.Categorical(cleaned, categories=categories)

        return df

    def prepare_matrices(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.Series]:
        """
        Split engineered DataFrame into feature matrix X, target series y, and timestamps.
        Excludes raw datetime, target, and intermediate string columns.
        """
        engineered = self.transform(df)

        exclude_cols = [
            self.timestamp_col,
            self.target_col,
            "pickup_date",
            "pickup_hour",
            "pickup_zone",
            "pickup_borough",
            "service_type",
        ]

        feature_cols = [c for c in engineered.columns if c not in exclude_cols]
        self.feature_columns = feature_cols

        X = engineered[feature_cols]
        y = engineered[self.target_col] if self.target_col in engineered.columns else None
        timestamps = engineered[self.timestamp_col]

        logger.info(f"Prepared feature matrix X: {X.shape}, target y: {len(y) if y is not None else 'None'}")
        return X, y, timestamps

    def prepare_train_val_split(
        self,
        df: pd.DataFrame,
        test_months: int = 2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split raw dataset using the Lookback Buffer pattern:
        1. Calculate cutoff timestamp once upfront.
        2. Partition df_train (<= cutoff) to fit feature engineer and transform independently.
        3. Create df_val_buffered starting (cutoff - max_lookback_hours) to compute continuous
           lags without artificial NaNs.
        4. Trim the buffer from validation matrices so evaluation strictly covers (> cutoff).
        Guarantees zero lookahead, zero target leakage, and zero redundant re-splits.
        """
        ts = pd.to_datetime(df[self.timestamp_col])
        max_date = ts.max()
        cutoff_date = max_date - pd.DateOffset(months=test_months)

        # Required lookback buffer: max of all lag hours and shifted rolling windows
        max_lag = max(self.lag_hours) if self.lag_hours else 24
        shift_horizon = min(self.lag_hours) if self.lag_hours else 24
        max_rolling = (max(self.rolling_windows) + shift_horizon) if self.rolling_windows else 48
        lookback_hours = max(max_lag, max_rolling)
        buffer_start = cutoff_date - pd.Timedelta(hours=lookback_hours)

        logger.info(
            f"Lookback Buffer Split: Cutoff={cutoff_date.date()} | "
            f"Buffer Lookback={lookback_hours}h (starts {buffer_start.date()})"
        )

        # 1. Isolate training and buffered validation datasets upfront
        df_train = df[ts <= cutoff_date].copy()
        df_val_buf = df[ts > buffer_start].copy()

        # 2. Fit feature engineer strictly on training set
        self.fit(df_train)

        # 3. Transform training features independently
        X_train, y_train, train_times = self.prepare_matrices(df_train)

        # 4. Transform buffered validation features
        X_val_buf, y_val_buf, val_times_buf = self.prepare_matrices(df_val_buf)

        # 5. Trim the lookback buffer so validation evaluation is strictly > cutoff_date
        val_mask = pd.to_datetime(val_times_buf) > cutoff_date
        X_val = X_val_buf[val_mask].copy().reset_index(drop=True)
        y_val = y_val_buf[val_mask].copy().reset_index(drop=True)

        logger.info(
            f"Lookback Buffer Split Complete: Train={len(X_train):,} rows ({train_times.min().date()} to {train_times.max().date()}) | "
            f"Val={len(X_val):,} rows (strictly post-cutoff: {cutoff_date.date()} onwards)"
        )
        return X_train, X_val, y_train, y_val

    def save(self, filepath: str) -> None:
        """Persist fitted engineer state and category mappings to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        artifacts = {
            "lag_hours": self.lag_hours,
            "rolling_windows": self.rolling_windows,
            "category_mappings": self.category_mappings,
            "feature_columns": self.feature_columns,
            "_is_fitted": self._is_fitted,
        }
        joblib.dump(artifacts, filepath)
        logger.info(f"Persisted feature engineering artifacts to {filepath}")

    def load(self, filepath: str) -> "TemporalFeatureEngineer":
        """Restore fitted engineer state from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature artifacts not found at {filepath}")
        artifacts = joblib.load(filepath)
        self.lag_hours = artifacts["lag_hours"]
        self.rolling_windows = artifacts["rolling_windows"]
        self.category_mappings = artifacts["category_mappings"]
        self.feature_columns = artifacts["feature_columns"]
        self._is_fitted = artifacts["_is_fitted"]
        logger.info(f"Loaded feature engineering artifacts from {filepath}")
        return self

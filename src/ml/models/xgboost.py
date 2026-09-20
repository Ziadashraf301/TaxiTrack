# src/ml/models/xgboost.py
"""
XGBoost Demand Forecaster (Strategy Pattern)
Concrete demand forecaster implemented using XGBoost GBDT regressor.
"""
import time
from typing import Dict, Optional, Any
import pandas as pd
from ml.models.base import BaseForecaster
from core.logging import get_logger

logger = get_logger(__name__)


class XGBoostForecaster(BaseForecaster):
    """Concrete demand forecaster implemented with XGBoost GBDT."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "enable_categorical": True,
        }
        if params:
            default_params.update(params)
        super().__init__(params=default_params)

    @property
    def name(self) -> str:
        return "XGBoost"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50,
    ) -> "XGBoostForecaster":
        """Train XGBoost regressor with native categorical support and early stopping."""
        from xgboost import XGBRegressor

        self.feature_names = X_train.columns.tolist()
        self.categorical_features = [
            c for c in self.feature_names if str(X_train[c].dtype) == "category"
        ]

        logger.info(
            f"Training XGBoost on {len(X_train):,} samples with {len(self.feature_names)} features. "
            f"Categorical features: {self.categorical_features}"
        )
        start_time = time.perf_counter()

        fit_params = dict(self.params)
        if X_val is not None and y_val is not None:
            fit_params["early_stopping_rounds"] = early_stopping_rounds

        self.model = XGBRegressor(**fit_params)
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=100,
        )

        self.training_duration_seconds = time.perf_counter() - start_time
        self.best_iteration_ = getattr(self.model, "best_iteration", self.params["n_estimators"])

        logger.info(
            f"XGBoost training completed in {self.training_duration_seconds:.2f}s. "
            f"Best iteration: {self.best_iteration_}"
        )
        return self

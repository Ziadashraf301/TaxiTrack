# src/ml/models/lightgbm.py
"""
LightGBM Demand Forecaster (Strategy Pattern)
Concrete demand forecaster implemented using LightGBM GBDT regressor.
"""
import time
from typing import Dict, Optional, Any
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor

from ml.models.base import BaseForecaster
from core.logging import get_logger

logger = get_logger(__name__)


class LightGBMForecaster(BaseForecaster):
    """Concrete demand forecaster implemented with LightGBM GBDT."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "n_estimators": 1500,
            "learning_rate": 0.05,
            "num_leaves": 128,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "objective": "regression",
            "metric": "mae",
            "importance_type": "gain",
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        super().__init__(params=default_params)

    @property
    def name(self) -> str:
        return "LightGBM"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50,
    ) -> "LightGBMForecaster":
        """Train LightGBM regressor with early stopping if validation data is provided."""
        self.feature_names = X_train.columns.tolist()
        self.categorical_features = [
            c for c in self.feature_names if str(X_train[c].dtype) == "category"
        ]

        logger.info(
            f"Training LightGBM on {len(X_train):,} samples with {len(self.feature_names)} features. "
            f"Categorical features: {self.categorical_features}"
        )
        start_time = time.perf_counter()
        self.model = LGBMRegressor(**self.params)

        callbacks = [lgb.log_evaluation(period=100)]
        eval_set = None

        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        self.training_duration_seconds = time.perf_counter() - start_time
        self.best_iteration_ = getattr(self.model, "best_iteration_", self.params["n_estimators"])

        logger.info(
            f"LightGBM training completed in {self.training_duration_seconds:.2f}s. "
            f"Best iteration: {self.best_iteration_}"
        )
        return self

# src/ml/models/base.py
"""
Abstract Forecaster Strategy & Evaluator Contracts (OOP & SOLID)
Defines extensible contracts for model fitting, non-negative inference, and evaluation.
"""
import os
import joblib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from core.logging import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Computes standard regression performance metrics: MAE, RMSE, WAPE, and R2."""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
        y_true_arr = np.array(y_true, dtype=float)
        y_pred_arr = np.array(y_pred, dtype=float)

        mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
        rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
        r2 = float(r2_score(y_true_arr, y_pred_arr))

        total_actual = float(np.sum(y_true_arr))
        wape = float(np.sum(np.abs(y_true_arr - y_pred_arr)) / max(total_actual, 1e-6)) * 100.0

        metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "wape_pct": round(wape, 2),
            "r2_score": round(r2, 4),
        }

        logger.info(
            f"{model_name} Evaluation -> MAE: {metrics['mae']} | RMSE: {metrics['rmse']} | "
            f"WAPE: {metrics['wape_pct']}% | R2: {metrics['r2_score']}"
        )
        return metrics


class BaseForecaster(ABC):
    """Abstract Strategy base class for demand forecasting models."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = params or {}
        self.model: Any = None
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.best_iteration_: Optional[int] = None
        self.training_duration_seconds: float = 0.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier of the forecaster architecture."""
        pass

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50,
    ) -> "BaseForecaster":
        """Train the model with optional early stopping."""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate non-negative demand predictions."""
        if self.model is None:
            raise RuntimeError(f"{self.name} model has not been trained yet. Call fit() or load() first.")

        if self.feature_names and list(X.columns) != self.feature_names:
            X = X[self.feature_names]

        raw_preds = self._raw_predict(X)
        return np.clip(raw_preds, a_min=0.0, a_max=None)

    def _raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Raw prediction delegate. Subclasses can override for framework-specific input handling."""
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Compute standard evaluation metrics."""
        preds = self.predict(X_test)
        return ModelEvaluator.evaluate(np.array(y_test), preds, model_name=self.name)

    def get_feature_importances(self) -> pd.DataFrame:
        """Extract and sort feature importances."""
        if self.model is None:
            raise RuntimeError(f"{self.name} model is not fitted.")

        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return pd.DataFrame(columns=["feature", "importance", "importance_pct"])

        total_imp = max(float(np.sum(importances)), 1e-9)
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
            "importance_pct": np.round((importances / total_imp) * 100.0, 2),
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, filepath: str) -> None:
        """Serialize model artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "categorical_features": self.categorical_features,
            "best_iteration": self.best_iteration_,
            "training_duration_seconds": self.training_duration_seconds,
        }
        joblib.dump(payload, filepath)
        logger.info(f"Saved {self.name} model artifact to {filepath}")

    def load(self, filepath: str) -> "BaseForecaster":
        """Deserialize model artifact from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        loaded = joblib.load(filepath)
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            self.params = loaded.get("params", self.params)
            self.feature_names = loaded.get("feature_names", [])
            self.categorical_features = loaded.get("categorical_features", [])
            self.best_iteration_ = loaded.get("best_iteration")
            self.training_duration_seconds = loaded.get("training_duration_seconds", 0.0)
        else:
            self.model = loaded
            self.feature_names = getattr(self.model, "feature_names_in_", getattr(self.model, "feature_name_", []))
        logger.info(f"Loaded {self.name} model from {filepath} ({len(self.feature_names)} features).")
        return self

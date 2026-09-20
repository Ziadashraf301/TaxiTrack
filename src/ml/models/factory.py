# src/ml/models/factory.py
"""
Forecaster Factory (Factory Pattern & Open/Closed Principle)
Instantiates demand forecasting model strategies dynamically based on configuration.
"""
from typing import Dict, Any, Optional
from ml.models.base import BaseForecaster
from ml.models.lightgbm import LightGBMForecaster
from ml.models.xgboost import XGBoostForecaster
from core.logging import get_logger

logger = get_logger(__name__)


class ForecasterFactory:
    """Factory creating concrete BaseForecaster instances."""

    _REGISTRY: Dict[str, type] = {
        "lightgbm": LightGBMForecaster,
        "xgboost": XGBoostForecaster,
    }

    @classmethod
    def register(cls, model_type: str, forecaster_cls: type) -> None:
        """Register a new forecaster architecture into the factory."""
        cls._REGISTRY[model_type.lower()] = forecaster_cls

    @classmethod
    def create(
        cls,
        model_type: str = "LightGBM",
        params: Optional[Dict[str, Any]] = None,
    ) -> BaseForecaster:
        """
        Instantiate and return a configured BaseForecaster strategy.
        """
        key = model_type.lower()
        forecaster_cls = cls._REGISTRY.get(key)

        if forecaster_cls is None:
            available = list(cls._REGISTRY.keys())
            raise ValueError(
                f"Unsupported model architecture '{model_type}'. Supported architectures: {available}"
            )

        logger.info(f"ForecasterFactory instantiated '{forecaster_cls.__name__}'")
        return forecaster_cls(params=params)

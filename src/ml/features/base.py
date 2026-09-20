# src/ml/features/base.py
"""
Abstract Feature Engineering Interface (Open/Closed Principle)
"""
from abc import ABC, abstractmethod
import pandas as pd


class BaseFeatureEngineer(ABC):
    """Abstract interface for temporal and tabular feature engineering."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseFeatureEngineer":
        """Fit feature encoders/scalers on training DataFrame."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into model-ready feature matrix."""
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in a single pass."""
        return self.fit(df).transform(df)

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persist feature engineering states/mappings."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> "BaseFeatureEngineer":
        """Restore feature engineering states/mappings."""
        pass

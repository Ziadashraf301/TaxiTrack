# src/ml/data/base.py
"""
Abstract Data Repository Contracts (Interface Segregation Principle)
Defines separated interfaces for demand forecasting data access and network graph metrics.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import pandas as pd


class BaseDemandRepository(ABC):
    """Abstract contract for querying demand forecasting datasets."""

    @abstractmethod
    def get_demand_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        service_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch demand time-series records."""
        pass

    @abstractmethod
    def get_demand_summary(self) -> Tuple[Optional[str], Optional[str], int]:
        """Fetch (min_date, max_date, total_records) summary."""
        pass


class BaseNetworkRepository(ABC):
    """Abstract contract for querying spatial transit network corridor metrics."""

    @abstractmethod
    def get_network_metrics(
        self,
        pickup_month: Optional[str] = None,
        min_trips: int = 10,
    ) -> pd.DataFrame:
        """Fetch origin-destination route volume metrics."""
        pass

# src/ml/graph/base.py
"""
Abstract Spatial Graph Analysis Interface (Interface Segregation Principle)
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class BaseGraphAnalyzer(ABC):
    """Abstract interface for spatial network and transit flow graph analysis."""

    @abstractmethod
    def build_graph(self, df: pd.DataFrame) -> Any:
        """Construct graph structure from origin-destination trip metrics."""
        pass

    @abstractmethod
    def compute_metrics(self) -> Dict[str, Any]:
        """Calculate network centralities, transit hubs, and flow bottlenecks."""
        pass

    @abstractmethod
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Return node-level network metrics as a tabular DataFrame."""
        pass

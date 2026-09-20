# src/ml/tracking/base.py
"""
Abstract Experiment Tracker Contract (Interface Segregation Principle)
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseExperimentTracker(ABC):
    """Abstract interface for experiment tracking and model registration."""

    @abstractmethod
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager starting a tracked run."""
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log configuration and hyperparameters."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log numerical performance metrics."""
        pass

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Upload file or directory as run artifact."""
        pass

    @abstractmethod
    def log_dataframe(self, df: pd.DataFrame, filename: str, artifact_path: Optional[str] = None) -> None:
        """Save and log a DataFrame artifact."""
        pass

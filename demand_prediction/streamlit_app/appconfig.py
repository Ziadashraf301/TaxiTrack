# ============================================================================
# FILE: app/config.py
# ============================================================================
"""
Configuration and constants for the application.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class AppConfig:
    """Application configuration constants."""
    PAGE_TITLE: str = "Taxi Demand Forecasting"
    PAGE_ICON: str = "🚕"
    LAYOUT: str = "wide"
    CACHE_TTL: int = 300  # seconds
    MIN_FORECAST_HORIZON: int = 1
    MAX_FORECAST_HORIZON: int = 1500
    DEFAULT_FORECAST_HORIZON: int = 2
    MIN_RECORD_COUNT: int = 24102
    RESULTS_DIR: str = "results"
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = 'localhost'
    port: int = 8123
    username: str = 'default'
    password: str = ''
    database: str = 'taxi_data'
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'DatabaseConfig':
        """Create from dictionary."""
        return cls(
            host=config.get('host', 'localhost'),
            port=config.get('port', 8123),
            username=config.get('username', 'default'),
            password=config.get('password', ''),
            database=config.get('database', 'taxi_data')
        )
"""FastAPI dependency injection providers for ClickHouse, cache, ML artifacts, and services."""
from typing import Optional
from cachetools import TTLCache
from fastapi import Depends, Request
from core.clickhouse import ClickHouseService
from api.services.analytics_service import AnalyticsService
from api.services.forecast_service import ForecastService
from api.services.network_service import NetworkService


def get_clickhouse(request: Request) -> ClickHouseService:
    """Provide a singleton ClickHouseService instance attached to app.state."""
    if not hasattr(request.app.state, "ch_service"):
        request.app.state.ch_service = ClickHouseService()
    return request.app.state.ch_service


def get_cache(request: Request) -> TTLCache:
    """Provide the in-memory TTLCache attached to app.state."""
    if not hasattr(request.app.state, "cache"):
        request.app.state.cache = TTLCache(maxsize=1024, ttl=300)
    return request.app.state.cache


def get_onnx_session(request: Request):
    """Provide the loaded ONNX Runtime InferenceSession if available."""
    return getattr(request.app.state, "onnx_session", None)


def get_feature_engineer(request: Request):
    """Provide the loaded TemporalFeatureEngineer instance if available."""
    return getattr(request.app.state, "feature_engineer", None)


def get_model_version(request: Request) -> str:
    """Return the active model version string."""
    return getattr(request.app.state, "model_version", "v1.0.0")


def get_analytics_service(
    ch: ClickHouseService = Depends(get_clickhouse),
    cache: TTLCache = Depends(get_cache),
) -> AnalyticsService:
    """Dependency provider for AnalyticsService."""
    return AnalyticsService(ch=ch, cache=cache)


def get_forecast_service(
    request: Request,
    ch: ClickHouseService = Depends(get_clickhouse),
    cache: TTLCache = Depends(get_cache),
) -> ForecastService:
    """Dependency provider for ForecastService."""
    return ForecastService(
        ch=ch,
        cache=cache,
        onnx_session=get_onnx_session(request),
        feature_engineer=get_feature_engineer(request),
        model_version=get_model_version(request),
    )


def get_network_service(
    ch: ClickHouseService = Depends(get_clickhouse),
    cache: TTLCache = Depends(get_cache),
) -> NetworkService:
    """Dependency provider for NetworkService."""
    return NetworkService(ch=ch, cache=cache)

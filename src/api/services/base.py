"""Base data service with generic cache-aside pattern and common ClickHouse utilities."""
from abc import ABC
from typing import Any, Callable, Optional
from cachetools import TTLCache
from core.clickhouse import ClickHouseService
from core.logging import get_logger

logger = get_logger(__name__)


class BaseDataService(ABC):
    """Abstract base service providing ClickHouse integration and caching facilities."""

    def __init__(self, ch: Optional[ClickHouseService] = None, cache: Optional[TTLCache] = None):
        self.ch = ch or ClickHouseService()
        self.cache = cache if cache is not None else TTLCache(maxsize=256, ttl=300)

    def _cached(self, key: str, fn: Callable[[], Any], ttl: Optional[int] = None) -> Any:
        """
        Generic cache-aside accessor. If key exists in cache, returns it immediately;
        otherwise executes fn(), caches the result, and returns it.
        """
        if self.cache is not None and key in self.cache:
            logger.debug(f"Cache HIT for key: {key}")
            return self.cache[key]

        logger.debug(f"Cache MISS for key: {key}. Executing query/computation...")
        result = fn()
        if self.cache is not None:
            self.cache[key] = result
        return result

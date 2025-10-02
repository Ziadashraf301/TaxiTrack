# ============================================================================
# FILE: app/database.py
# ============================================================================
"""
Database operations and queries.
"""

import logging
from typing import Dict, Optional
import pandas as pd
import clickhouse_connect

logger = logging.getLogger(__name__)


class DatabaseClient:
    """Handles all database operations."""
    
    def __init__(self, config: Dict):
        """
        Initialize database client.
        
        Args:
            config: Database configuration dictionary
        """
        self.config = config
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of database client."""
        if self._client is None:
            try:
                self._client = clickhouse_connect.get_client(
                    host=self.config.get('host', 'localhost'),
                    port=self.config.get('port', 8123),
                    username=self.config.get('username', 'default'),
                    password=self.config.get('password', ''),
                    database=self.config.get('database', 'taxi_data')
                )
                logger.info("Database client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize database client: {e}")
                raise
        return self._client
    
    def fetch_available_groups(self, min_records: int = 16068) -> pd.DataFrame:
        """
        Fetch available forecast groups from database.
        
        Args:
            min_records: Minimum number of records required for a group
            
        Returns:
            DataFrame with pickup_borough, pickup_zone, service_type
            
        Raises:
            Exception: If query fails
        """
        query = f"""
        SELECT pickup_borough, pickup_zone, service_type
        FROM mart_demand_prediction
        GROUP BY pickup_borough, pickup_zone, service_type
        HAVING count(*) >= {min_records}
        ORDER BY count(*) DESC
        """
        
        try:
            df = self.client.query_df(query)
            logger.info(f"Fetched {len(df)} available groups from database")
            return df
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client.query("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self._client is not None:
            try:
                self._client.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")
            finally:
                self._client = None
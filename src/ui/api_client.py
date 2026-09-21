"""Typed HTTP client for communicating with the TaxiTrack FastAPI backend."""
import os
from typing import Any, Dict, List, Optional
import httpx


class APIClientError(Exception):
    """Raised when an API request returns an error or fails to connect."""
    pass


class TaxiTrackClient:
    """Synchronous HTTP client for Streamlit UI with robust error handling and timeout defaults."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 15.0):
        self.base_url = (base_url or os.getenv("API_BASE_URL", "http://localhost:8000")).rstrip("/")
        self.timeout = timeout

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{endpoint}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 404:
                    raise APIClientError(f"Resource not found: {resp.json().get('detail', '404 Not Found')}")
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError:
            raise APIClientError(
                f"Cannot connect to TaxiTrack API at {self.base_url}. Ensure the FastAPI server is running."
            )
        except httpx.HTTPStatusError as e:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            raise APIClientError(f"API Error ({e.response.status_code}): {detail}")
        except Exception as e:
            raise APIClientError(f"Unexpected error calling {url}: {str(e)}")

    def get_health(self) -> Dict[str, Any]:
        """Fetch API health status and loaded model version."""
        return self._get("/health")

    def get_overview(
        self,
        start_date: str,
        end_date: str,
        borough: Optional[str] = None,
        service_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve high-level operational KPIs and period-over-period delta values."""
        params = {"start_date": start_date, "end_date": end_date}
        if borough and borough.lower() != "all":
            params["borough"] = borough
        if service_type and service_type.lower() != "all":
            params["service_type"] = service_type
        return self._get("/api/analytics/overview", params=params)

    def get_timeseries(
        self,
        start_date: str,
        end_date: str,
        granularity: str = "daily",
        borough: Optional[str] = None,
        service_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch aggregated trip volume and revenue timeseries."""
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "granularity": granularity,
        }
        if borough and borough.lower() != "all":
            params["borough"] = borough
        if service_type and service_type.lower() != "all":
            params["service_type"] = service_type
        return self._get("/api/analytics/timeseries", params=params)

    def get_breakdown(
        self,
        start_date: str,
        end_date: str,
        borough: Optional[str] = None,
        service_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch borough and service type cross-tabulation table data."""
        params = {"start_date": start_date, "end_date": end_date}
        if borough and borough.lower() != "all":
            params["borough"] = borough
        if service_type and service_type.lower() != "all":
            params["service_type"] = service_type
        return self._get("/api/analytics/breakdown", params=params)

    def get_historical(
        self,
        start_date: str,
        end_date: str,
        pickup_zone: str,
        pickup_borough: str,
        service_type: str,
    ) -> List[Dict[str, Any]]:
        """Fetch historical observed hourly demand points."""
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "pickup_zone": pickup_zone,
            "pickup_borough": pickup_borough,
            "service_type": service_type,
        }
        return self._get("/api/forecast/historical", params=params)

    def get_zones(self) -> List[Dict[str, Any]]:
        """Fetch all available boroughs and zones from the backend warehouse."""
        return self._get("/api/forecast/zones")

    def get_forecast(
        self,
        pickup_zone: str,
        pickup_borough: str,
        service_type: str,
        horizon_hours: int = 720,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate forward-looking hourly demand predictions using ONNX inference."""
        params = {
            "pickup_zone": pickup_zone,
            "pickup_borough": pickup_borough,
            "service_type": service_type,
            "horizon_hours": horizon_hours,
        }
        if end_date:
            params["end_date"] = end_date
        return self._get("/api/forecast/predict", params=params)

    def get_top_corridors(
        self,
        pickup_month: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        zone: Optional[str] = None,
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fetch top origin-destination transit corridors."""
        params: Dict[str, Any] = {"top_n": top_n}
        if pickup_month:
            params["pickup_month"] = pickup_month
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if zone and zone.lower() != "all":
            params["zone"] = zone
        return self._get("/api/network/top-corridors", params=params)

    def get_centrality(
        self,
        pickup_month: str,
    ) -> List[Dict[str, Any]]:
        """Fetch NetworkX spatial graph centrality metrics."""
        params = {"pickup_month": pickup_month}
        return self._get("/api/network/centrality", params=params)

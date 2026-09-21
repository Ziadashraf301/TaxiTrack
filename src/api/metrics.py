"""Prometheus business metrics definitions for TaxiTrack Serving API."""
from prometheus_client import Counter, Gauge, Histogram, Info

# Business telemetry metrics
FORECAST_REQUESTS = Counter(
    "taxitrack_forecast_requests_total",
    "Total ONNX demand forecast inferences requested",
    ["zone", "service_type"],
)

FORECAST_LATENCY = Histogram(
    "taxitrack_forecast_latency_seconds",
    "Time spent computing ONNX demand inference in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

DRIFT_ALERT_GAUGE = Gauge(
    "taxitrack_drift_alert",
    "Data drift detection alert status (1=drift detected, 0=healthy)",
)

MODEL_VERSION_INFO = Info(
    "taxitrack_model_version",
    "Current active demand forecaster model version and metadata",
)

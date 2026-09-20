# src/ml/pipeline.py
"""
End-to-End ML Training Pipeline Orchestrator (Facade Pattern)
Orchestrates: data load -> feature engineering -> LightGBM training -> MLflow log -> ONNX export.
"""

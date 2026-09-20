# src/core/logging.py
"""
Centralized Logging Configuration
Emits structured logs seamlessly captured by Airflow Task UI, Docker, and CLI execution.
"""
import logging
import os
import sys


def get_logger(name: str = "taxitrack", level: int = logging.INFO) -> logging.Logger:
    """
    Get or configure a structured logger.
    Seamlessly integrates with Airflow's task logging UI while providing
    clean stdout logging for standalone CLI, tests, and microservices.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if running under Airflow or if root logger already has active handlers
    root_has_handlers = bool(logging.getLogger().handlers)
    in_airflow_context = "AIRFLOW_HOME" in os.environ or "AIRFLOW_CTX_DAG_ID" in os.environ

    if in_airflow_context or root_has_handlers:
        # Running inside Airflow task execution context:
        # Propagate cleanly to Airflow's task logger hierarchy (FileTaskHandler).
        # This prevents duplicate log entries in the Airflow UI.
        logger.propagate = True
    elif not logger.handlers:
        # Standalone context (FastAPI, CLI, local script, tests):
        # Attach a dedicated StreamHandler to stdout with custom formatting.
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


# src/data/ingestors/factory.py
"""
Ingestor Factory (Factory Pattern)
Dynamically instantiates dataset-specific ingestors.
"""
from core.logging import get_logger
from data.ingestors.base import BaseIngestor
from data.ingestors.green import GreenTaxiIngestor
from data.ingestors.yellow import YellowTaxiIngestor

logger = get_logger(__name__)


class IngestorFactory:
    """Factory creating concrete BaseIngestor instances based on dataset type."""

    @staticmethod
    def get_ingestor(dataset_type: str) -> BaseIngestor:
        normalized = dataset_type.lower().strip()
        if "green" in normalized:
            ingestor = GreenTaxiIngestor()
        elif "yellow" in normalized:
            ingestor = YellowTaxiIngestor()
        else:
            logger.error(f"Failed to resolve ingestor: unsupported dataset type '{dataset_type}'.")
            raise ValueError(f"Unsupported dataset type: '{dataset_type}'. Expected 'green' or 'yellow'.")

        logger.debug(f"Factory created {ingestor.__class__.__name__} for dataset_type='{dataset_type}'")
        return ingestor

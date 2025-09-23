import logging
import sys

def setup_logging(log_file="app.log"):
    logger = logging.getLogger("TaxiTrackLogger")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
        )

        # Console handler -> goes to stdout (captured by Airflow)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # Propagate logs to Airflow root logger
        logger.propagate = True

        logger.info("Custom logging is set up.")

    return logger

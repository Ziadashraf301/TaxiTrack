import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Setup and return a logger with both console and optional file handlers.

    Args:
        name (str): Logger name (usually __name__).
        log_file (str, optional): Path to log file. If None, only console logging is used.
        level (int): Logging level (default=logging.INFO).

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # Prevent adding handlers multiple times

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (UTF-8 safe)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional, UTF-8 safe)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
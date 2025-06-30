import os
import logging
from pathlib import Path


def setup_logger(logger_name: str ="vec_tool_test", save_logs: bool =True) -> logging.Logger:
    """Configure up logging for tests."""
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if save_logs:
        log_dir = Path(os.path.dirname(os.path.realpath(__file__))) / "../../logs/"
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{logger_name}.log")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

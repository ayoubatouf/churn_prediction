import logging
from pathlib import Path


def setup_logger(log_file_name: Path) -> None:

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_dir = log_file_name.parent

    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info("Logging setup complete.")

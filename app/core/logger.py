import logging

from app.core.config import LOG_LEVEL


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    return logging.getLogger("dataset_research")
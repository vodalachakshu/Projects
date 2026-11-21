"""Structured logging setup for the Document Q&A RAG Service.""[span_36](start_span)"[span_36](end_span)

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root and application loggers.""[span_37](start_span)"[span_37](end_span)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)

    app_logger = logging.getLogger("rag_service")
    app_logger.setLevel(numeric_level)
    return app_logger

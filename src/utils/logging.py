"""
Structured logging setup using structlog with human-readable console output or JSON.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.config import LoggingConfig


_BANNER = """
  ╔══════════════════════════════════════╗
  ║   Polymarket Market Maker  v1.0.0   ║
  ╚══════════════════════════════════════╝
"""


def print_banner() -> None:
    print(_BANNER, flush=True)


def setup_logging(cfg: LoggingConfig) -> None:
    """Configure structlog and stdlib logging together."""
    log_path = Path(cfg.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, cfg.level)

    # Console handler uses human-readable format;
    # file handler always uses JSON for machine parsing.
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.handlers.RotatingFileHandler(
        cfg.file,
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
        encoding="utf-8",
    )

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=[console_handler, file_handler],
    )

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "asyncio", "websockets", "web3", "eth_account", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Console always uses the readable renderer; file always uses JSON.
    console_renderer = structlog.dev.ConsoleRenderer(
        colors=True,
        exception_formatter=structlog.dev.plain_traceback,
    )
    json_renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )

    # Console handler → human-readable
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=console_renderer,
        foreign_pre_chain=shared_processors,
    )
    console_handler.setFormatter(console_formatter)

    # File handler → JSON regardless of config
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=json_renderer,
        foreign_pre_chain=shared_processors,
    )
    file_handler.setFormatter(file_formatter)


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)

"""
Logging utilities (moved to `utils.log`).
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from functools import wraps
import time
import json


DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'

    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            log_message = f"{self.COLORS[record.levelname]}{log_message}{self.RESET}"
        return log_message


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        return json.dumps(log_data, ensure_ascii=False)


class ContextLogger(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger, context: Dict[str, Any] = None):
        super().__init__(logger, context or {})
    def process(self, msg, kwargs):
        context_str = ' '.join(f'{k}={v}' for k, v in self.extra.items())
        if context_str:
            msg = f"[{context_str}] {msg}"
        return msg, kwargs
    def with_context(self, **kwargs) -> 'ContextLogger':
        new_context = {**self.extra, **kwargs}
        return ContextLogger(self.logger, new_context)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    max_bytes: int = 10485760,
    backup_count: int = 5,
    colored_console: bool = True,
    json_file: bool = False
) -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers.clear()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = ColoredFormatter(log_format, date_format) if colored_console else logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = JSONFormatter() if json_file else logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    return root_logger


def get_logger(name: str, context: Dict[str, Any] = None) -> ContextLogger:
    logger = logging.getLogger(name)
    return ContextLogger(logger, context)


def log_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.debug(f"{func.__name__} executed in {duration:.4f}s")
        return result
    return wrapper


def log_exceptions(logger: logging.Logger = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _logger.exception(f"Exception in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


class LogCapture:
    def __init__(self, logger_name: str = None, level: int = logging.DEBUG):
        self.logger_name = logger_name
        self.level = level
        self.messages: list = []
        self._handler = None
    def __enter__(self):
        class CaptureHandler(logging.Handler):
            def __init__(self, messages):
                super().__init__()
                self.messages = messages
            def emit(self, record):
                self.messages.append({'level': record.levelname, 'message': self.format(record), 'logger': record.name, 'timestamp': datetime.fromtimestamp(record.created)})
        self._handler = CaptureHandler(self.messages)
        self._handler.setLevel(self.level)
        self._handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        logger = logging.getLogger(self.logger_name)
        logger.addHandler(self._handler)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = logging.getLogger(self.logger_name)
        logger.removeHandler(self._handler)
        return False
    def get_messages(self, level: str = None) -> list:
        if level:
            return [m for m in self.messages if m['level'] == level.upper()]
        return self.messages


def init_default_logging():
    try:
        setup_logging(level="INFO", log_file="logs/app.log", colored_console=True)
    except Exception:
        logging.basicConfig(level=logging.INFO, format=DEFAULT_FORMAT)


init_default_logging()

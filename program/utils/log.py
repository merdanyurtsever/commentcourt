"""
Logging Utilities for CommentCourt.

This module provides centralized logging configuration with:
- Console and file handlers
- Rotating log files
- Colored console output (optional)
- Performance logging
- Context-aware logging
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


# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter.
    
    Adds ANSI color codes to log messages based on level.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_message = super().format(record)
        
        if record.levelname in self.COLORS:
            log_message = f"{self.COLORS[record.levelname]}{log_message}{self.RESET}"
        
        return log_message


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.
    
    Outputs log records as JSON objects for easy parsing.
    """
    
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
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        
        return json.dumps(log_data, ensure_ascii=False)


class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter with context support.
    
    Allows adding context information to all log messages.
    """
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any] = None):
        super().__init__(logger, context or {})
    
    def process(self, msg, kwargs):
        # Add context to message
        context_str = ' '.join(f'{k}={v}' for k, v in self.extra.items())
        if context_str:
            msg = f"[{context_str}] {msg}"
        return msg, kwargs
    
    def with_context(self, **kwargs) -> 'ContextLogger':
        """Create a new logger with additional context."""
        new_context = {**self.extra, **kwargs}
        return ContextLogger(self.logger, new_context)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    colored_console: bool = True,
    json_file: bool = False
) -> logging.Logger:
    """
    Setup application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Log message format
        date_format: Date format for timestamps
        max_bytes: Max size of log file before rotation
        backup_count: Number of backup files to keep
        colored_console: Use colored console output
        json_file: Use JSON format for file logging
        
    Returns:
        Root logger
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    if colored_console:
        console_formatter = ColoredFormatter(log_format, date_format)
    else:
        console_formatter = logging.Formatter(log_format, date_format)
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        if json_file:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(log_format, date_format)
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str, context: Dict[str, Any] = None) -> ContextLogger:
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name (usually __name__)
        context: Context data to include in messages
        
    Returns:
        ContextLogger instance
    """
    logger = logging.getLogger(name)
    return ContextLogger(logger, context)


def log_performance(func):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_performance
        def my_function():
            ...
    """
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
    """
    Decorator to log exceptions.
    
    Usage:
        @log_exceptions()
        def my_function():
            ...
    """
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
    """
    Context manager to capture log messages.
    
    Useful for testing or capturing logs for display.
    
    Usage:
        with LogCapture() as capture:
            # ... code that logs ...
        
        messages = capture.messages
    """
    
    def __init__(self, logger_name: str = None, level: int = logging.DEBUG):
        """
        Initialize log capture.
        
        Args:
            logger_name: Name of logger to capture (None for root)
            level: Minimum level to capture
        """
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
                self.messages.append({
                    'level': record.levelname,
                    'message': self.format(record),
                    'logger': record.name,
                    'timestamp': datetime.fromtimestamp(record.created)
                })
        
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
        """Get captured messages, optionally filtered by level."""
        if level:
            return [m for m in self.messages if m['level'] == level.upper()]
        return self.messages


class ProgressLogger:
    """
    Logger for tracking progress of long-running operations.
    
    Usage:
        progress = ProgressLogger(total=100, logger=logger)
        for i in range(100):
            # ... work ...
            progress.update()
    """
    
    def __init__(self, total: int, logger: logging.Logger = None,
                 prefix: str = "Progress", log_interval: int = 10):
        """
        Initialize progress logger.
        
        Args:
            total: Total number of items
            logger: Logger to use
            prefix: Prefix for log messages
            log_interval: Percentage interval for logging
        """
        self.total = total
        self.logger = logger or logging.getLogger(__name__)
        self.prefix = prefix
        self.log_interval = log_interval
        
        self.current = 0
        self.last_logged_percent = 0
        self.start_time = time.time()
    
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        
        percent = int((self.current / self.total) * 100)
        
        if percent >= self.last_logged_percent + self.log_interval or self.current == self.total:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            self.logger.info(
                f"{self.prefix}: {self.current}/{self.total} ({percent}%) "
                f"- {rate:.1f} items/s - ETA: {eta:.0f}s"
            )
            
            self.last_logged_percent = percent
    
    def finish(self) -> None:
        """Log completion."""
        elapsed = time.time() - self.start_time
        rate = self.total / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            f"{self.prefix}: Completed {self.total} items in {elapsed:.1f}s ({rate:.1f} items/s)"
        )


# Initialize default logging on module import
_initialized = False


def init_default_logging():
    """Initialize default logging configuration."""
    global _initialized
    if not _initialized:
        setup_logging(
            level="INFO",
            log_file="logs/app.log",
            colored_console=True
        )
        _initialized = True


# Auto-initialize with safe defaults
try:
    init_default_logging()
except Exception:
    # Fallback to basic config if initialization fails
    logging.basicConfig(level=logging.INFO, format=DEFAULT_FORMAT)

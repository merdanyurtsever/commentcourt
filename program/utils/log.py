# REMOVED: program.utils.log shim deleted (contents removed during migration)

# This file was a compatibility shim and has been cleared to remove
# legacy code. Import logging utilities from `utils.log` or use the
# canonical logging setup in `utils/log.py`.
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

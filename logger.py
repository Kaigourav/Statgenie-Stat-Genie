"""
Structured Logging Module
Provides JSON-formatted logging with context and request tracking.
"""
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from config import config


class JSONFormatter(logging.Formatter):
    """Custom formatter for structured JSON logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        if hasattr(record, "job_id"):
            log_data["job_id"] = record.job_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        
        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter"""
    
    def __init__(self):
        fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')


def setup_logging():
    """Initialize application logging based on config"""
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.logging.LOG_LEVEL.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Choose formatter
    if config.logging.LOG_FORMAT == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if config.logging.LOG_FILE:
        try:
            file_handler = logging.FileHandler(config.logging.LOG_FILE)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to create log file handler: {e}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module"""
    return logging.getLogger(name)


# Initialize logging on import
app_logger = setup_logging()

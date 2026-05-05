"""
Logging utilities for eval system
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


_logger = None
_llm_logger = None


def setup_logging(log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging for eval system
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Logger instance
    """
    global _logger, _llm_logger
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create main logger
    logger = logging.getLogger("eval")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler - for flow logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler - for flow logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eval_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    _logger = logger
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    # Create LLM logger - only file, no console
    llm_logger = logging.getLogger("eval.llm")
    llm_logger.setLevel(logging.DEBUG)
    llm_logger.handlers.clear()
    llm_logger.propagate = False  # Don't propagate to parent (no console output)
    
    llm_log_file = log_dir / f"llm_{timestamp}.log"
    llm_file_handler = logging.FileHandler(llm_log_file, encoding='utf-8')
    llm_file_handler.setLevel(logging.DEBUG)
    llm_file_format = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    llm_file_handler.setFormatter(llm_file_format)
    llm_logger.addHandler(llm_file_handler)
    
    _llm_logger = llm_logger
    logger.info(f"LLM logging initialized. Log file: {llm_log_file}")
    
    return logger


def get_logger() -> logging.Logger:
    """Get the global logger instance"""
    global _logger
    if _logger is None:
        _logger = logging.getLogger("eval")
        if not _logger.handlers:
            # Basic console handler if not initialized
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
    return _logger


def get_llm_logger() -> logging.Logger:
    """Get the LLM logger instance (file-only, for raw model I/O)"""
    global _llm_logger
    if _llm_logger is None:
        _llm_logger = logging.getLogger("eval.llm")
        _llm_logger.setLevel(logging.DEBUG)
        _llm_logger.propagate = False
    return _llm_logger

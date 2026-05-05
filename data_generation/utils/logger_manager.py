"""
Centralized logger management system
Provides a single global logger for the entire data generation system
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


class LoggerManager:
    """
    Singleton logger manager - provides ONE global logger for entire system
    
    Features:
    - Single logger instance shared across all modules
    - Console output + optional file logging
    - Prevents duplicate handlers
    """
    
    _instance: Optional['LoggerManager'] = None
    _initialized: bool = False
    _logger: Optional[logging.Logger] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if LoggerManager._initialized:
            return
        
        self._log_dir: Optional[Path] = None
        self._default_level = logging.INFO
        self._log_file: Optional[Path] = None
        self._file_handler: Optional[logging.FileHandler] = None
        LoggerManager._initialized = True
    
    def configure(
        self, 
        log_dir: Path, 
        level: int = logging.INFO,
        enable_file: bool = True
    ):
        """
        Configure the global logger
        
        Args:
            log_dir: Directory to save log files
            level: Logging level
            enable_file: Whether to enable file logging
        """
        self._log_dir = log_dir
        self._default_level = level
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the single global logger
        if LoggerManager._logger is None:
            LoggerManager._logger = self._create_logger(enable_file)
    
    def _create_logger(self, enable_file: bool) -> logging.Logger:
        """Create the global logger instance"""
        logger = logging.getLogger("data_generation")
        logger.setLevel(self._default_level)
        logger.propagate = False
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Console handler - suppress all logger output, only print() shows on console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.CRITICAL)
        logger.addHandler(console_handler)
        
        # File handler - timestamped log file
        if enable_file and self._log_dir:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._log_file = self._log_dir / f"run_{timestamp}.log"
            self._file_handler = logging.FileHandler(self._log_file, mode='w', encoding='utf-8')
            self._file_handler.setLevel(self._default_level)
            self._file_handler.setFormatter(file_formatter)
            logger.addHandler(self._file_handler)
        
        return logger
    
    def get_logger(self) -> logging.Logger:
        """
        Get the global logger instance
        
        Returns:
            The global logger
        """
        if LoggerManager._logger is None:
            raise RuntimeError("LoggerManager not configured. Call configure() first.")
        return LoggerManager._logger
    
    def clear(self):
        """Clear logger (useful for testing)"""
        if LoggerManager._logger:
            for handler in LoggerManager._logger.handlers[:]:
                handler.close()
                LoggerManager._logger.removeHandler(handler)
        LoggerManager._logger = None
        LoggerManager._initialized = False
    
    @classmethod
    def reset(cls):
        """Reset singleton instance (useful for testing)"""
        if cls._instance:
            cls._instance.clear()
        cls._instance = None
        cls._initialized = False
    
    def log_llm_interaction(self, agent_name: str, role: str, content: str, label: str = ""):
        """
        Log LLM interaction to file only (not console)
        
        Args:
            agent_name: Name of the agent (Expert, RedTeam, etc.)
            role: SYSTEM/HUMAN/AI/INPUT/OUTPUT
            content: The message content
            label: Optional label for context
        """
        if self._file_handler is None or self._log_file is None:
            return
        
        separator = "=" * 60
        header = f"[{agent_name}] [{role}]" + (f" - {label}" if label else "")
        
        # Write directly to file (bypass console) - no timestamp prefix for cleaner format
        with open(self._log_file, 'a', encoding='utf-8') as f:
            f.write(f"{separator}\n{header}\n{separator}\n{content}\n{separator}\n\n")


# Global singleton instance
logger_manager = LoggerManager()


# Convenience functions
def configure_logging(log_dir: Path, level: int = logging.INFO, enable_file: bool = True):
    """
    Configure the global logger
    
    Args:
        log_dir: Directory to save log files
        level: Logging level
        enable_file: Whether to enable file logging
    """
    logger_manager.configure(log_dir, level, enable_file)


def get_logger() -> logging.Logger:
    """
    Get the global logger instance
    
    Returns:
        The global logger
    """
    return logger_manager.get_logger()


def log_llm_interaction(agent_name: str, role: str, content: str, label: str = ""):
    """
    Log LLM interaction to main log
    
    Args:
        agent_name: Name of the agent
        role: SYSTEM/HUMAN/AI
        content: Message content
        label: Optional label
    """
    logger_manager.log_llm_interaction(agent_name, role, content, label)

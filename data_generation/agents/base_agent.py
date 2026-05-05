"""
Base Agent class for all AI agents in the system

Provides common functionality:
- LLM initialization with standard configuration
- Logger access
- Temperature and model configuration
- Retry mechanism for API errors
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableSerializable
from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from config.settings import config
from utils.logger_manager import get_logger, log_llm_interaction

# Get logger for retry logging
_retry_logger = logging.getLogger("llm_retry")


class LoggingLLMWrapper(RunnableSerializable):
    """Wrapper that logs all LLM interactions to file while maintaining Runnable interface"""
    
    _llm: BaseChatModel
    _agent_name: str
    
    def __init__(self, llm: BaseChatModel, agent_name: str, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '_llm', llm)
        object.__setattr__(self, '_agent_name', agent_name)
    
    @property
    def InputType(self):
        return self._llm.InputType
    
    @property
    def OutputType(self):
        return self._llm.OutputType
    
    def invoke(self, input: Union[str, List[BaseMessage]], config=None, **kwargs):
        """Invoke LLM and log the interaction (file only, not console)"""
        # Format input for logging
        if isinstance(input, str):
            input_str = input
        elif isinstance(input, list):
            input_str = "\n---\n".join([
                f"[{msg.__class__.__name__}]: {msg.content}"
                for msg in input
            ])
        else:
            input_str = str(input)
        
        # Log input to file only
        log_llm_interaction(self._agent_name, "INPUT", input_str)
        
        # Call actual LLM with retry
        response = self._invoke_with_retry(input, config, **kwargs)
        
        # Log output to file only
        log_llm_interaction(self._agent_name, "OUTPUT", response.content)
        
        return response
    
    @retry(
        retry=retry_if_exception_type((
            RateLimitError,
            APIError,
            APITimeoutError,
            APIConnectionError,
        )),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=before_sleep_log(_retry_logger, logging.WARNING),
        reraise=True,
    )
    def _invoke_with_retry(self, input, config=None, **kwargs):
        """Invoke LLM with retry logic for transient errors"""
        return self._llm.invoke(input, config=config, **kwargs)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped LLM"""
        return getattr(self._llm, name)


class BaseAgent(ABC):
    """
    Base class for all AI agents
    
    Provides:
    - Standard LLM initialization
    - Logger instance
    - Configuration management
    """
    
    def __init__(
        self,
        agent_name: str,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize base agent
        
        Args:
            agent_name: Name of the agent (for logging)
            temperature: LLM temperature (default from config)
            model: LLM model name (default from config)
            **kwargs: Additional arguments for subclasses
        """
        self.agent_name = agent_name
        self.logger = get_logger()
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.model = model or config.LLM_MODEL
        
        # Initialize LLM with logging wrapper
        self.llm = self._create_llm()
        
        self.logger.debug(f"[{self.agent_name}] Initialized with model={self.model}, temperature={self.temperature}")
    
    def _create_llm(self) -> LoggingLLMWrapper:
        """
        Create LLM instance wrapped with logging
        
        Returns:
            LoggingLLMWrapper that logs all interactions to file
        """
        llm_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": config.LLM_MAX_TOKENS,
            "api_key": config.OPENAI_API_KEY,
            "max_retries": 5,
            "timeout": 300,
            "request_timeout": 300,
        }
        
        if config.OPENAI_BASE_URL:
            llm_kwargs["base_url"] = config.OPENAI_BASE_URL

        raw_llm = ChatOpenAI(**llm_kwargs)
        return LoggingLLMWrapper(raw_llm, self.agent_name)
    
    @abstractmethod
    def get_agent_type(self) -> str:
        """
        Get agent type identifier
        
        Returns:
            Agent type string (e.g., "generator", "tester", "analyzer")
        """
        pass
    
    def log_info(self, message: str):
        """Log info message with agent prefix"""
        self.logger.info(f"[{self.agent_name}] {message}")
    
    def log_debug(self, message: str):
        """Log debug message with agent prefix"""
        self.logger.debug(f"[{self.agent_name}] {message}")
    
    def log_error(self, message: str):
        """Log error message with agent prefix"""
        self.logger.error(f"[{self.agent_name}] {message}")


class GenerationAgent(BaseAgent):
    """
    Base class for code/test generation agents
    
    Adds:
    - Work directory management
    - Toolbox for code operations
    """
    
    def __init__(
        self,
        agent_name: str,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        work_dir: Optional[Path] = None,
        **kwargs
    ):
        """
        Initialize generation agent
        
        Args:
            agent_name: Name of the agent
            temperature: LLM temperature
            model: LLM model name
            work_dir: Working directory for file operations
            **kwargs: Additional arguments
        """
        super().__init__(agent_name, temperature, model, **kwargs)
        self.work_dir = work_dir
        if self.work_dir:
            self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def get_agent_type(self) -> str:
        return "generator"


class EvaluationAgent(BaseAgent):
    """
    Base class for evaluation/analysis agents
    
    Focused on analyzing code, tests, or failures
    """
    
    def get_agent_type(self) -> str:
        return "evaluator"


class OrchestrationAgent(BaseAgent):
    """
    Base class for orchestration/coordination agents
    
    Manages workflows and coordinates other agents
    """
    
    def get_agent_type(self) -> str:
        return "orchestrator"

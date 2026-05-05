"""
Base agent class for eval system
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Literal, Callable, Any
import logging
import time
from functools import wraps

import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from config.settings import config
from utils.logger import get_llm_logger

# Import transformers LLM (lazy import to avoid dependency issues)
try:
    from .transformers_llm import TransformersLLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Common thinking tag patterns used by reasoning models
THINKING_TAG_PATTERNS = [
    r'<think>.*?</think>',
    r'<thinking>.*?</thinking>',
    r'<reasoning>.*?</reasoning>',
    r'<reflection>.*?</reflection>',
    r'<inner_thought>.*?</inner_thought>',
]


def retry_on_api_error(max_retries: int = 2, retry_delay: float = 1.0):
    """
    Decorator for retrying LLM invocations when API/network errors occur.
    This handles infrastructure-level errors (e.g., 502, timeout, connection errors).
    Does NOT retry on application-level errors (parsing, validation).
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            last_exception = None
            
            # Save the original message history before first attempt
            original_messages = self.get_messages().copy() if hasattr(self, 'get_messages') else None
            
            for attempt in range(max_retries + 1):
                try:
                    # Restore original messages for retry (don't accumulate retry prompts)
                    if attempt > 0 and original_messages is not None:
                        self.chat_history.messages = original_messages.copy()
                    
                    result = func(self, *args, **kwargs)
                    return result
                    
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()
                    
                    # Only retry on API/network errors, not parsing errors
                    is_api_error = any(keyword in error_str for keyword in [
                        '502', '503', '504', 'timeout', 'connection', 'network',
                        'rate limit', 'too many requests', 'service unavailable'
                    ])
                    
                    if attempt < max_retries and is_api_error:
                        self.logger.warning(
                            f"API error on attempt {attempt + 1}/{max_retries + 1}: {str(e)}. "
                            f"Retrying in {retry_delay}s..."
                        )
                        time.sleep(retry_delay)
                    else:
                        if not is_api_error:
                            self.logger.error(f"Non-retryable error: {str(e)}")
                        else:
                            self.logger.error(
                                f"All {max_retries + 1} attempts failed. Last error: {str(e)}"
                            )
                        break
            
            # If all retries failed, raise the last exception
            raise last_exception
        
        return wrapper
    return decorator


# Keep old name for backward compatibility but mark as deprecated
def retry_on_output_error(max_retries: int = 2, retry_delay: float = 1.0):
    """
    DEPRECATED: Use retry_on_api_error instead.
    This decorator is kept for backward compatibility.
    """
    return retry_on_api_error(max_retries, retry_delay)


class BaseAgent(ABC):
    """Base class for all eval agents"""
    
    def __init__(
        self,
        agent_name: str,
        model: str,
        temperature: float,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        engine: Literal["openai", "transformers"] = "openai",
        max_timeout: int = 120,
        **engine_kwargs
    ):
        """
        Initialize base agent
        
        Args:
            agent_name: Name of the agent
            model: Model name
            temperature: Temperature setting
            api_key: API key (optional, uses config if not provided)
            base_url: Base URL (optional, uses config if not provided)
            engine: LLM engine type ("openai" or "transformers")
            max_timeout: Max timeout for API requests in seconds (default: 120)
            **engine_kwargs: Additional engine-specific parameters
        """
        self.agent_name = agent_name
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or config.OPENAI_API_KEY
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.engine = engine
        self.engine_kwargs = engine_kwargs
        self.max_timeout = max_timeout
        
        self.logger = logging.getLogger(f"eval.{agent_name}")
        
        # Chat history for multi-turn conversations
        self.chat_history = InMemoryChatMessageHistory()
        
        # Create LLM
        self.llm = self._create_llm()
    
    def _create_llm(self):
        """Create LLM instance based on engine type"""
        if self.engine == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "Transformers engine requires 'transformers' and 'torch' packages. "
                    "Install with: pip install transformers torch accelerate"
                )
            
            llm_kwargs = {
                "model_name": self.model,
                "temperature": self.temperature,
                "max_tokens": config.TARGET_MAX_TOKENS,
                **self.engine_kwargs
            }
            
            self.logger.info(f"Creating Transformers LLM with model: {self.model}")
            return TransformersLLM(**llm_kwargs)
        
        else:  # Default to OpenAI-compatible API
            llm_kwargs = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": config.LLM_MAX_TOKENS,
                "api_key": self.api_key,
                "timeout": self.max_timeout,
                "max_retries": self._get_max_retries(),
            }
            
            if self.base_url:
                llm_kwargs["base_url"] = self.base_url
            
            return ChatOpenAI(**llm_kwargs)
    
    def _get_max_retries(self) -> int:
        """
        Get max retries for this agent type
        
        Returns:
            Max retries for API calls
        """
        agent_type = self.get_agent_type()
        if agent_type == "collaborator":
            return getattr(config, "COLLABORATOR_MAX_RETRIES", 2)
        elif agent_type == "target":
            return getattr(config, "TARGET_MAX_RETRIES", 2)
        else:
            return 2  # Default for other agents
    
    def reset_history(self):
        """Reset chat history"""
        self.chat_history = InMemoryChatMessageHistory()
    
    def add_system_message(self, content: str):
        """Add system message to history"""
        self.chat_history.add_message(SystemMessage(content=content))
    
    def add_human_message(self, content: str):
        """Add human message to history"""
        self.chat_history.add_message(HumanMessage(content=content))
    
    def add_ai_message(self, content: str):
        """Add AI message to history"""
        self.chat_history.add_message(AIMessage(content=content))
    
    def get_messages(self) -> List[BaseMessage]:
        """Get all messages in history"""
        return self.chat_history.messages
    
    def invoke(self, messages: List[BaseMessage] = None) -> str:
        """
        Invoke LLM with messages
        
        Args:
            messages: Messages to send (uses history if None)
            
        Returns:
            LLM response content
        """
        if messages is None:
            messages = self.get_messages()
        
        # Log input to LLM logger (file only)
        llm_logger = get_llm_logger()
        llm_logger.info(f"\n{'='*60}")
        llm_logger.info(f"[{self.agent_name}] LLM INVOKE - Model: {self.model}")
        llm_logger.info(f"{'='*60}")
        llm_logger.info("--- INPUT MESSAGES ---")
        for i, msg in enumerate(messages):
            msg_type = msg.__class__.__name__
            llm_logger.info(f"[{i}] {msg_type}:")
            llm_logger.info(msg.content)
            llm_logger.info("-" * 40)
        
        # Invoke LLM
        response = self.llm.invoke(messages)
        
        # Log output
        llm_logger.info("--- OUTPUT ---")
        llm_logger.info(response.content)
        llm_logger.info(f"{'='*60}\n")
        
        return response.content
    
    def clear_cache(self):
        """
        Clear GPU cache if using transformers engine
        Should be called after each test case to prevent memory overflow
        """
        if self.engine == "transformers" and hasattr(self.llm, 'clear_cache'):
            self.llm.clear_cache()
    
    def _build_retry_prompt(self, error_msg: str) -> str:
        """
        Build a retry prompt to guide the model after a parsing failure.
        
        Args:
            error_msg: The error message from the failed attempt
            
        Returns:
            Retry prompt message
        """
        return f"""Your previous response had a formatting issue: {error_msg}

Please provide your response again, ensuring it follows the exact format specified in the instructions.
Pay special attention to:
1. JSON format (if applicable): proper quotes, brackets, and commas
2. Code blocks: proper markdown formatting with language tags
3. Complete and valid output

Please try again:"""
    
    @abstractmethod
    def get_agent_type(self) -> str:
        """Get agent type identifier"""
        pass
    
    def remove_thinking_content(self, response: str) -> str:
        """
        Remove thinking/reasoning content from model response.
        This handles reasoning models like Claude, DeepSeek, QwQ, etc.
        
        Args:
            response: Raw model response
            
        Returns:
            Response with thinking tags removed
        """
        cleaned = response
        for pattern in THINKING_TAG_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

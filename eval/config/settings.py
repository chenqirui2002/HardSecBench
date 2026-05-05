"""
Eval system configuration
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """Eval system configuration"""
    
    # Project paths
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    EVAL_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    # Output paths
    OUTPUT_DIR: Path = field(default=None)
    LOGS_DIR: Path = field(default=None)
    
    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    DISABLE_PROXY: bool = True
    
    # Target LLM Engine Configuration
    TARGET_ENGINE: str = "openai"  # "openai" or "transformers"
    TARGET_DEVICE: str = "cuda"  # For transformers: "cuda", "cpu", or "auto"
    TARGET_BATCH_SIZE: int = 1  # For transformers: batch size for inference
    TARGET_MAX_TOKENS: int = 4096  # For transformers engine: max tokens to generate for local HF inference
    TARGET_LOAD_IN_8BIT: bool = False  # For transformers: use 8-bit quantization
    TARGET_LOAD_IN_4BIT: bool = False  # For transformers: 4-bit quantization
    TARGET_TORCH_DTYPE: str = "auto"  # For transformers: "auto", "float16", "bfloat16", "float32"
    
    # Collaborator configuration
    COLLABORATOR_MODEL: str = "gpt-5"
    COLLABORATOR_TEMPERATURE: float = 0.3
    COLLABORATOR_API_KEY: Optional[str] = None
    COLLABORATOR_BASE_URL: Optional[str] = None
    COLLABORATOR_TIMEOUT: int = 600  # Timeout for collaborator API calls in seconds
    COLLABORATOR_MAX_RETRIES: int = 2  # Max retries for collaborator API calls
    
    # LLM settings
    LLM_MAX_TOKENS: int = 40000  # For OpenAI-compatible API calls: passed to ChatOpenAI(max_tokens=...)
    TARGET_TIMEOUT: int = 600  # Timeout for target LLM API calls in seconds
    TARGET_MAX_RETRIES: int = 2  # Max retries for target LLM API calls
    
    # Logging Configuration
    LLM_LOGGING_ENABLED: bool = True  # Enable/disable detailed LLM I/O logging
    
    # LangSmith Configuration
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "HardSecBench-Eval"
    
    def __post_init__(self):
        # Set default paths
        if self.OUTPUT_DIR is None:
            self.OUTPUT_DIR = self.EVAL_ROOT / "outputs"
        if self.LOGS_DIR is None:
            self.LOGS_DIR = self.OUTPUT_DIR / "logs"
        
        # Create directories
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load from environment
        self._load_from_env()
        
        # Disable proxy
        self._disable_proxy()
        
        # Setup LangSmith
        self._setup_langsmith()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Try to load from eval .env first
        eval_env_file = self.EVAL_ROOT / ".env"
        if eval_env_file.exists():
            self._load_env_file(eval_env_file)
        else:
            # Fallback to data_generation .env
            data_gen_env_file = self.PROJECT_ROOT / "data_generation" / ".env"
            if data_gen_env_file.exists():
                self._load_env_file(data_gen_env_file)
        
        # Override with environment variables
        if os.getenv("OPENAI_API_KEY"):
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_BASE_URL"):
            self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
        
        # Target engine config
        if os.getenv("TARGET_ENGINE"):
            self.TARGET_ENGINE = os.getenv("TARGET_ENGINE")
        if os.getenv("TARGET_DEVICE"):
            self.TARGET_DEVICE = os.getenv("TARGET_DEVICE")
        if os.getenv("TARGET_BATCH_SIZE"):
            self.TARGET_BATCH_SIZE = int(os.getenv("TARGET_BATCH_SIZE"))
        if os.getenv("TARGET_MAX_TOKENS"):
            self.TARGET_MAX_TOKENS = int(os.getenv("TARGET_MAX_TOKENS"))
        if os.getenv("TARGET_LOAD_IN_8BIT"):
            self.TARGET_LOAD_IN_8BIT = os.getenv("TARGET_LOAD_IN_8BIT", "false").lower() == "true"
        if os.getenv("TARGET_LOAD_IN_4BIT"):
            self.TARGET_LOAD_IN_4BIT = os.getenv("TARGET_LOAD_IN_4BIT", "false").lower() == "true"
        if os.getenv("TARGET_TORCH_DTYPE"):
            self.TARGET_TORCH_DTYPE = os.getenv("TARGET_TORCH_DTYPE")
        
        # Collaborator config
        if os.getenv("COLLABORATOR_MODEL"):
            self.COLLABORATOR_MODEL = os.getenv("COLLABORATOR_MODEL")
        if os.getenv("COLLABORATOR_API_KEY"):
            self.COLLABORATOR_API_KEY = os.getenv("COLLABORATOR_API_KEY")
        if os.getenv("COLLABORATOR_BASE_URL"):
            self.COLLABORATOR_BASE_URL = os.getenv("COLLABORATOR_BASE_URL")
        
        # LangSmith config
        if os.getenv("LANGCHAIN_TRACING_V2"):
            self.LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        if os.getenv("LANGCHAIN_API_KEY"):
            self.LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
        # Logging config
        if os.getenv("LLM_LOGGING_ENABLED"):
            self.LLM_LOGGING_ENABLED = os.getenv("LLM_LOGGING_ENABLED", "true").lower() == "true"
        
        if os.getenv("LANGCHAIN_PROJECT"):
            self.LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
        if os.getenv("LANGCHAIN_ENDPOINT"):
            self.LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
    
    def _load_env_file(self, env_file: Path):
        """Load .env file"""
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    
                    # Target LLM config
                    if key == "OPENAI_API_KEY" and not self.OPENAI_API_KEY:
                        self.OPENAI_API_KEY = value
                    elif key == "OPENAI_BASE_URL" and not self.OPENAI_BASE_URL:
                        self.OPENAI_BASE_URL = value
                    
                    # Target engine config
                    elif key == "TARGET_ENGINE":
                        self.TARGET_ENGINE = value
                    elif key == "TARGET_DEVICE":
                        self.TARGET_DEVICE = value
                    elif key == "TARGET_BATCH_SIZE":
                        self.TARGET_BATCH_SIZE = int(value)
                    elif key == "TARGET_MAX_TOKENS":
                        self.TARGET_MAX_TOKENS = int(value)
                    elif key == "TARGET_LOAD_IN_8BIT":
                        self.TARGET_LOAD_IN_8BIT = value.lower() == "true"
                    elif key == "TARGET_LOAD_IN_4BIT":
                        self.TARGET_LOAD_IN_4BIT = value.lower() == "true"
                    elif key == "TARGET_TORCH_DTYPE":
                        self.TARGET_TORCH_DTYPE = value
                    
                    # Collaborator config
                    elif key == "COLLABORATOR_MODEL":
                        self.COLLABORATOR_MODEL = value
                    elif key == "COLLABORATOR_API_KEY" and not self.COLLABORATOR_API_KEY:
                        self.COLLABORATOR_API_KEY = value
                    elif key == "COLLABORATOR_BASE_URL" and not self.COLLABORATOR_BASE_URL:
                        self.COLLABORATOR_BASE_URL = value
                    
                    # LangSmith config
                    elif key == "LANGCHAIN_TRACING_V2":
                        self.LANGCHAIN_TRACING_V2 = value.lower() == "true"
                    elif key == "LANGCHAIN_API_KEY" and not self.LANGCHAIN_API_KEY:
                        self.LANGCHAIN_API_KEY = value
                    elif key == "LANGCHAIN_PROJECT":
                        self.LANGCHAIN_PROJECT = value
                    # Logging config
                    elif key == "LLM_LOGGING_ENABLED":
                        self.LLM_LOGGING_ENABLED = value.lower() == "true"
                    
                    elif key == "LANGCHAIN_ENDPOINT":
                        self.LANGCHAIN_ENDPOINT = value
    
    def _disable_proxy(self):
        """Disable proxy settings if configured"""
        if not self.DISABLE_PROXY:
            return
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
        for var in proxy_vars:
            if var in os.environ:
                del os.environ[var]
    
    def _setup_langsmith(self):
        """Setup LangSmith tracing"""
        if self.LANGCHAIN_TRACING_V2 and self.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = self.LANGCHAIN_ENDPOINT
            os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT


# Global config instance
config = EvalConfig()

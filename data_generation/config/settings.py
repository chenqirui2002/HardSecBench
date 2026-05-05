"""
System configuration
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class SystemConfig(BaseSettings):
    """System-wide configuration"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.resolve()
    DATA_GEN_ROOT: Path = PROJECT_ROOT / "data_generation"
    
    # Input data paths
    DIMENSIONS_PATH: Path = DATA_GEN_ROOT / "dimensions.json"
    CWE_DATABASE_PATH: Path = PROJECT_ROOT / "hardware_design_cwe" / "hardware_design_cwe.json"
    
    # Output paths
    OUTPUT_DIR: Path = DATA_GEN_ROOT / "outputs"
    LOGS_DIR: Path = OUTPUT_DIR / "logs"
    
    # LLM Configuration
    LLM_PROVIDER: str = "openai"

    LLM_MODEL: str = "gpt-5" 
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 40000
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    DISABLE_PROXY: bool = True
    
    # LangSmith Configuration (for monitoring and debugging)
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"  # US: api.smith.langchain.com, EU: eu.smith.langchain.com
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "HardSecBench"
    
    # Agent Configuration (None = use LLM_MODEL)
    # Agent 0: Seed Generator (uses Architect config)
    
    # Agent A: Architect
    ARCHITECT_MODEL: Optional[str] = None  # None means use LLM_MODEL
    ARCHITECT_TEMPERATURE: float = 0.8
    
    # Agent B: Expert
    EXPERT_MODEL: Optional[str] = None  # None means use LLM_MODEL
    EXPERT_TEMPERATURE: float = 0.2
    
    # Agent C: Red Teamer
    RED_TEAMER_MODEL: Optional[str] = None  # None means use LLM_MODEL
    RED_TEAMER_TEMPERATURE: float = 0.4
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Disable proxy if configured
        if self.DISABLE_PROXY:
            proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
            for var in proxy_vars:
                if var in os.environ:
                    del os.environ[var]
        
        # Create output directories
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load API key from environment if not set
        if not self.OPENAI_API_KEY:
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        # Load base URL from environment if not set
        if not self.OPENAI_BASE_URL:
            self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
        
        # Load LangSmith configuration from environment if not set
        if os.getenv("LANGCHAIN_TRACING_V2"):
            self.LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        if not self.LANGCHAIN_API_KEY:
            self.LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
        if os.getenv("LANGCHAIN_PROJECT"):
            self.LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
        if os.getenv("LANGCHAIN_ENDPOINT"):
            self.LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
        
        # Set LangSmith environment variables for LangChain to use
        if self.LANGCHAIN_TRACING_V2 and self.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = self.LANGCHAIN_ENDPOINT
            os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT
            print(f"✓ LangSmith tracing enabled for project: {self.LANGCHAIN_PROJECT}")


# Global config instance
config = SystemConfig()

"""
Transformers local LLM implementation
Provides a LangChain-compatible interface for local transformers models
"""
from typing import Optional, List, Any, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from config.settings import config


class TransformersLLM(BaseChatModel):
    """
    LangChain-compatible wrapper for local transformers models
    Supports chat-based models with proper message formatting
    """
    
    model_name: str
    model: Any = None
    tokenizer: Any = None
    device: str = "cuda"
    temperature: float = 0.7
    batch_size: int = None  # Will be set from config.TARGET_BATCH_SIZE if None
    max_tokens: int = None  # Will be set from config.TARGET_MAX_TOKENS if None
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    
    # Model loading parameters
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "auto"
    trust_remote_code: bool = True
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set batch_size from config if not provided
        if self.batch_size is None:
            self.batch_size = config.TARGET_BATCH_SIZE
        # Set max_tokens from config if not provided
        if self.max_tokens is None:
            self.max_tokens = config.TARGET_MAX_TOKENS
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer"""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            
            # Determine torch dtype
            if self.torch_dtype == "auto":
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            elif self.torch_dtype == "float16":
                dtype = torch.float16
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            
            # Load tokenizer with better error handling
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    use_fast=False  # Use slow tokenizer to avoid conversion issues
                )
            except Exception as e:
                print(f"Warning: Failed to load tokenizer with use_fast=False: {e}")
                print("Trying with use_fast=True...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    use_fast=True
                )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization if specified
            model_kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "dtype": dtype,
            }
            
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
            elif self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = self.device if torch.cuda.is_available() else "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
    
    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """
        Format messages for the model
        Supports common chat templates
        """
        # Try to use the model's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            # Convert LangChain messages to dict format
            chat_messages = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    chat_messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, HumanMessage):
                    chat_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    chat_messages.append({"role": "assistant", "content": msg.content})
            
            try:
                formatted = self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}")
                # Fall back to manual formatting
        
        # Manual formatting for models without chat template
        formatted_parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_parts.append(f"System: {msg.content}\n")
            elif isinstance(msg, HumanMessage):
                formatted_parts.append(f"User: {msg.content}\n")
            elif isinstance(msg, AIMessage):
                formatted_parts.append(f"Assistant: {msg.content}\n")
        
        formatted_parts.append("Assistant:")
        return "\n".join(formatted_parts)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from messages"""
        # Format messages
        prompt = self._format_messages(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tokens
        )
        
        # Apply batch size
        if self.batch_size > 1:
            # Duplicate inputs for batch processing
            input_ids = inputs['input_ids'].repeat(self.batch_size, 1)
            attention_mask = inputs['attention_mask'].repeat(self.batch_size, 1)
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        # Move to device
        if not self.load_in_8bit and not self.load_in_4bit:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generation config
        gen_config = GenerationConfig(
            max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            top_k=kwargs.get("top_k", self.top_k),
            repetition_penalty=kwargs.get("repetition_penalty", self.repetition_penalty),
            do_sample=kwargs.get("do_sample", self.do_sample),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                return_dict_in_generate=True,
            )
        
        # Decode
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Create chat generation
        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
    
    def clear_cache(self):
        """
        Clear GPU cache to prevent memory overflow
        Should be called after each test case in sequential mode
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM"""
        return "transformers"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "batch_size": self.batch_size,
            "max_tokens": self.max_tokens,
        }
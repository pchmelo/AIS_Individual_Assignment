"""
Abstract Base Model Client

This module defines the interface that all model clients must implement.
Extend this class to add support for new AI providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a model's capabilities."""
    name: str
    provider: str
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_json_output: bool = False
    supports_structured_output: bool = False
    supports_streaming: bool = False
    max_tokens: int = 4096
    context_window: int = 8192


class BaseModelClient(ABC):
    """
    Abstract base class for all model clients.
    
    To create a new client:
    1. Inherit from this class
    2. Implement all abstract methods
    3. Call super().__init__() in your constructor
    4. Register the client in ClientFactory
    
    Example:
        class MyCustomClient(BaseModelClient):
            def __init__(self, model: str, api_key: str):
                super().__init__()
                self.model = model
                self.api_key = api_key
            
            def generate(self, messages, temperature=0.2, max_tokens=4096):
                # Implementation here
                pass
            
            def get_model_info(self):
                return ModelInfo(
                    name=self.model,
                    provider="my_provider",
                    supports_function_calling=True
                )
    """
    
    def __init__(self):
        """Initialize base client attributes."""
        self._initialized = False
    
    @abstractmethod
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.2, 
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """
        Generate a response from the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles can be 'system', 'user', or 'assistant'.
            temperature: Sampling temperature (0.0 to 1.0). Lower = more deterministic.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional provider-specific parameters.
        
        Returns:
            The generated text response.
        
        Raises:
            Exception: If API call fails or model is unavailable.
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the model's capabilities.
        
        Returns:
            ModelInfo dataclass with model capabilities.
        """
        pass
    
    def supports_function_calling(self) -> bool:
        """
        Check if the model supports function/tool calling.
        
        Returns:
            True if function calling is supported.
        """
        return self.get_model_info().supports_function_calling
    
    def supports_vision(self) -> bool:
        """
        Check if the model supports vision/image inputs.
        
        Returns:
            True if vision is supported.
        """
        return self.get_model_info().supports_vision
    
    def supports_streaming(self) -> bool:
        """
        Check if the model supports streaming responses.
        
        Returns:
            True if streaming is supported.
        """
        return self.get_model_info().supports_streaming
    
    def get_provider(self) -> str:
        """
        Get the provider name for this client.
        
        Returns:
            Provider name string.
        """
        return self.get_model_info().provider
    
    def get_model_name(self) -> str:
        """
        Get the model name.
        
        Returns:
            Model name string.
        """
        return self.get_model_info().name
    
    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """
        Validate message format before sending to model.
        
        Args:
            messages: List of message dictionaries.
        
        Returns:
            True if messages are valid.
        
        Raises:
            ValueError: If messages are invalid.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        valid_roles = {'system', 'user', 'assistant'}
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dictionary")
            if 'role' not in msg:
                raise ValueError(f"Message {i} missing 'role' key")
            if 'content' not in msg:
                raise ValueError(f"Message {i} missing 'content' key")
            if msg['role'] not in valid_roles:
                raise ValueError(f"Message {i} has invalid role '{msg['role']}'. Must be one of {valid_roles}")
        
        return True
    
    def _build_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Build a simple text prompt from messages (for models without chat API).
        
        Args:
            messages: List of message dictionaries.
        
        Returns:
            Formatted prompt string.
        """
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant:"
        return prompt
    
    def __repr__(self) -> str:
        info = self.get_model_info()
        return f"{self.__class__.__name__}(model='{info.name}', provider='{info.provider}')"

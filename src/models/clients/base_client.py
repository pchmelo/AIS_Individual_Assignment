from abc import ABC, abstractmethod
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class ModelInfo:
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
    """
    
    def __init__(self):
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
        Returns:
            The generated text response by the model.  
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the model's capabilities.
        """
        pass
    
    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """
        Validate message format before sending to model.
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

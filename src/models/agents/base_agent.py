from abc import ABC, abstractmethod
from typing import List, Dict
from models.clients.base_client import BaseModelClient
from models.clients.local_client import LocalModelClient

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Agents are responsible for:
    - Processing user messages
    - Interacting with model clients
    - Optionally using tools
    """
    
    def __init__(
        self, 
        model_client: BaseModelClient = None, 
        model_name: str = None
    ):
        """
        Initialize base agent.
        
        Args:
            model_client: Pre-configured model client instance
            model_name: Model name for creating local client (if model_client is None)
        """
        if model_client is None:
            model_name = model_name or "ibm-granite/granite-3b-code-instruct"
            self.model_client = LocalModelClient(model=model_name)
        else:
            self.model_client = model_client
    
    def ask_model(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.2, 
        max_tokens: int = 4096
    ) -> str:
        """
        Send messages to the model and get a response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Model response text
        """
        return self.model_client.generate(messages, temperature, max_tokens)
    
    @abstractmethod
    def run(self, user_message: str) -> str:
        """
        Process a user message and return a response.
        
        Args:
            user_message: The user's input message
        
        Returns:
            Agent's response
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        
        Returns:
            System prompt string
        """
        pass
    
    def get_model_info(self) -> dict:
        """Get information about the model being used."""
        info = self.model_client.get_model_info()
        return {
            "name": info.name,
            "provider": info.provider,
            "supports_function_calling": info.supports_function_calling
        }

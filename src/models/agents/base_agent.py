from abc import ABC, abstractmethod
from typing import List, Dict
from models.clients.base_client import BaseModelClient
from models.clients.local_client import LocalModelClient

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """
    def __init__(
        self, 
        model_client: BaseModelClient = None, 
        model_name: str = None
    ):
        if model_client is None:
            model_name = model_name
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
        """
        return self.model_client.generate(messages, temperature, max_tokens)
    
    @abstractmethod
    def run(self, user_message: str) -> str:
        """
        Process a user message and return a response.
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        """
        pass

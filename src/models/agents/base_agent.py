from abc import ABC, abstractmethod
from typing import List, Dict
import time
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
        max_tokens: int = 4096,
        max_retries: int = 3
    ) -> str:
        """
        Send messages to the model and get a response.
        Returns None if the model fails to respond after retries.
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                result = self.model_client.generate(messages, temperature, max_tokens)
                if result:
                    return result
                print(f"Empty response on attempt {attempt + 1}/{max_retries}")
            except Exception as e:
                last_error = e
                print(f"Error in ask_model (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Retrying in {delay}s...")
                time.sleep(delay)
        
        if last_error:
            print(f"All {max_retries} attempts failed. Last error: {last_error}")
        return None
    
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

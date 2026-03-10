from abc import ABC, abstractmethod
from typing import List, Dict
import time
import re
from models.clients.base_client import BaseModelClient
from models.clients.local_client import LocalModelClient


class APIError(Exception):
    """Exception raised for API errors with user-friendly messages."""
    
    ERROR_MESSAGES = {
        401: "Authentication failed. Check your API key is correct and active.",
        403: "Access forbidden. Your API key may not have permission for this model.",
        429: "Rate limit exceeded. Please wait and try again.",
        500: "Server error. The API provider is experiencing issues.",
        502: "Bad gateway. The API provider is temporarily unavailable.",
        503: "Service unavailable. The API provider is overloaded.",
    }
    
    def __init__(self, message: str, status_code: int = None):
        self.status_code = status_code
        self.original_message = message
        
        # Parse status code from message if not provided
        if status_code is None:
            match = re.search(r'(\d{3})', message)
            if match:
                status_code = int(match.group(1))
                self.status_code = status_code
        
        # Get user-friendly message
        friendly = self.ERROR_MESSAGES.get(status_code, "")
        if friendly:
            super().__init__(f"{friendly}")
        else:
            super().__init__(message)

    def __str__(self):
        return super().__str__()

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
        Raises APIError if the model fails to respond after retries.
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                result = self.model_client.generate(messages, temperature, max_tokens)
                if result:
                    return result
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check for non-retryable errors (auth issues)
                if "401" in error_str or "403" in error_str:
                    raise APIError(error_str)
            
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                time.sleep(delay)
        
        # All retries exhausted
        if last_error:
            raise APIError(str(last_error))
        raise APIError("Model returned empty response after all retries.")
    
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

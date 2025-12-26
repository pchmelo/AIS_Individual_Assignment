from abc import ABC, abstractmethod
from agents.model_client import BaseModelClient, LocalModelClient

class BaseAgent(ABC):    
    def __init__(self, model_client: BaseModelClient = None, model_name: str = None):
        if model_client is None:
            model_name = model_name or "ibm-granite/granite-3b-code-instruct"
            self.model_client = LocalModelClient(model_name)
        else:
            self.model_client = model_client
    
    def ask_model(self, messages, temperature=0.2, max_tokens=4096):
        return self.model_client.generate(messages, temperature, max_tokens)
    
    @abstractmethod
    def run(self, user_message: str) -> str:
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        pass
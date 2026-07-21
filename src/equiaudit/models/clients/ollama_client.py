import requests
from typing import List, Dict, Any
from models.clients.base_client import BaseModelClient, ModelInfo

class OllamaClient(BaseModelClient):
    """
    Client for local Ollama API.
    Ollama runs locally on the user's machine on port 11434 by default.
    """
    
    DEFAULT_HOST = "http://localhost:11434"
    
    def __init__(
        self, 
        model: str = None,
        host: str = None,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.host = host or self.DEFAULT_HOST
        self._initialized = True
        
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.2, 
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        self.validate_messages(messages)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
        }
        
        # Pass optional args specifically handled by ollama if needed
        if "top_p" in kwargs:
            payload["options"]["top_p"] = kwargs["top_p"]
        
        try:
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=kwargs.get("timeout", 180))
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API Error: {str(e)}")
            
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model,
            provider="ollama",
            supports_vision=False,
            supports_function_calling=True,
            supports_json_output=True,
            supports_structured_output=True,
            supports_streaming=False,
            max_tokens=4096,
            context_window=8192
        )

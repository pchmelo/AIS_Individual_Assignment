"""
OpenRouter API Client

Provides access to various models through the OpenRouter API.
Supports models from OpenAI, Anthropic, Google, Meta, and others.
"""

import os
import time
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from models.clients.base_client import BaseModelClient, ModelInfo

load_dotenv()


class OpenRouterClient(BaseModelClient):
    """
    Client for OpenRouter API.
    
    OpenRouter provides unified access to multiple AI models through a single API.
    
    Environment Variables:
        OPENROUTER_API_KEY: Your OpenRouter API key (required)
    
    Example:
        client = OpenRouterClient(model="x-ai/grok-4.1-fast:free")
        response = client.generate([
            {"role": "user", "content": "Hello!"}
        ])
    """
    
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "x-ai/grok-4.1-fast:free"
    
    def __init__(
        self, 
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        model_info: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize OpenRouter client.
        
        Args:
            model: Model identifier (e.g., "x-ai/grok-4.1-fast:free")
            base_url: API base URL (defaults to OpenRouter API)
            api_key: API key (defaults to OPENROUTER_API_KEY env var)
            model_info: Optional dict with model capabilities
            **kwargs: Additional configuration options
        """
        super().__init__()
        
        self.model = model or self.DEFAULT_MODEL
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        # Model capabilities info
        self._model_info = model_info or {}
        
        # HTTP headers for requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": kwargs.get("referer", "http://localhost:3000"),
            "X-Title": kwargs.get("title", "AI Agent")
        }
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )
        
        self._initialized = True
        print(f"OpenRouter client initialized: {self.model}")
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.2, 
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """
        Generate a response using the OpenRouter API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters (top_p, stop, etc.)
        
        Returns:
            Generated text response.
        
        Raises:
            Exception: If API call fails.
        """
        self.validate_messages(messages)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        
        retryable_codes = {429, 500, 502, 503, 504}
        max_retries = kwargs.get("max_retries", 5)
        base_delay = 2  # seconds
        timeout = kwargs.get("timeout", 120)

        for attempt in range(max_retries + 1):
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=timeout,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]

            if response.status_code in retryable_codes and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(
                    f"OpenRouter {response.status_code} (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay}s..."
                )
                time.sleep(delay)
                continue

            raise Exception(f"OpenRouter API Error: {response.status_code} - {response.text}")
        
        # Should never reach here, but just in case
        raise Exception(f"OpenRouter API Error after {max_retries + 1} attempts: {response.status_code} - {response.text}")
    
    def get_model_info(self) -> ModelInfo:
        """Get model capabilities info."""
        return ModelInfo(
            name=self.model,
            provider="openrouter",
            supports_vision=self._model_info.get("vision", False),
            supports_function_calling=self._model_info.get("function_calling", True),
            supports_json_output=self._model_info.get("json_output", False),
            supports_structured_output=self._model_info.get("structured_output", True),
            supports_streaming=True,
            max_tokens=self._model_info.get("max_tokens", 4096),
            context_window=self._model_info.get("context_window", 128000)
        )

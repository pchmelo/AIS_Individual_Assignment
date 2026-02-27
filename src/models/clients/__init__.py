"""
Model Clients Module

This module provides abstract and concrete implementations of model clients
for different AI providers (local, OpenRouter, Gemini, etc.).

To add a new client:
1. Create a new file (e.g., my_client.py)
2. Inherit from BaseModelClient
3. Implement required abstract methods
4. Register in client_factory.py
"""

from models.clients.base_client import BaseModelClient
from models.clients.openrouter_client import OpenRouterClient
from models.clients.gemini_client import GeminiClient
from models.clients.local_client import LocalModelClient
from models.clients.client_factory import ClientFactory

__all__ = [
    'BaseModelClient',
    'OpenRouterClient', 
    'GeminiClient',
    'LocalModelClient',
    'ClientFactory'
]

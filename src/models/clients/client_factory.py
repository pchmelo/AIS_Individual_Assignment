from typing import Dict, Any, Type
from models.clients.base_client import BaseModelClient
from models.clients.openrouter_client import OpenRouterClient
from models.clients.gemini_client import GeminiClient
from models.clients.ollama_client import OllamaClient

class ClientFactory:
    """
    Factory for creating model clients.
    """
    
    _providers: Dict[str, Type[BaseModelClient]] = None
    
    @classmethod
    def _ensure_providers_loaded(cls):
        if cls._providers is None:            
            cls._providers = {
                "openrouter": OpenRouterClient,
                "gemini": GeminiClient,
                "google": GeminiClient,
                "ollama": OllamaClient,
            }
    
    @classmethod
    def create(
        cls,
        provider: str,
        model: str = None,
        **kwargs
    ) -> BaseModelClient:
        """
        Create a model client for the specified provider.
        """

        cls._ensure_providers_loaded()
        provider_lower = provider.lower()
        
        if provider_lower not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider: '{provider}'. "
                f"Available providers: {available}"
            )
        
        client_class = cls._providers[provider_lower]
        
        if model:
            kwargs["model"] = model
        
        return client_class(**kwargs)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseModelClient:
        """
        Create a client from a configuration dictionary.
        """
        config = config.copy()
        provider = config.pop("provider", None)
        
        if not provider:
            raise ValueError("Config must include 'provider' key")
        
        return cls.create(provider, **config)
    
    @classmethod
    def from_yaml_config(cls, yaml_config: Dict[str, Any], model_name: str = None) -> BaseModelClient:
        """
        Create a client from a YAML configuration structure.
        """
        models_config = yaml_config.get("models")
        if models_config is not None:
            if model_name is None:
                model_name = yaml_config.get("default_model")
                if model_name is None:
                    model_name = next(iter(models_config))

            model_cfg = models_config.get(model_name)
            if model_cfg is None:
                available = ", ".join(models_config.keys())
                raise ValueError(
                    f"Model '{model_name}' not found in config. "
                    f"Available models: {available}"
                )
            model_cfg = model_cfg.copy()
            provider = model_cfg.pop("provider", model_name)
            return cls.from_config({"provider": provider, **model_cfg})

        raise ValueError("Config must have a 'models' section")

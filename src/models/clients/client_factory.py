"""
Client Factory

Factory for creating model clients based on configuration.
This allows easy switching between different AI providers.
"""

from typing import Dict, Any, Optional, Type
from models.clients.base_client import BaseModelClient


# Registry of available client classes
_CLIENT_REGISTRY: Dict[str, Type[BaseModelClient]] = {}


def register_client(name: str):
    """
    Decorator to register a client class in the factory.
    
    Example:
        @register_client("my_provider")
        class MyProviderClient(BaseModelClient):
            ...
    """
    def decorator(cls: Type[BaseModelClient]):
        _CLIENT_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_registered_clients() -> Dict[str, Type[BaseModelClient]]:
    """Get dictionary of all registered client classes."""
    return _CLIENT_REGISTRY.copy()


class ClientFactory:
    """
    Factory for creating model clients.
    
    Supports creating clients from:
    - Provider name (openrouter, gemini, local)
    - Configuration dictionary
    - Full config file
    
    Example:
        # Create by provider name
        client = ClientFactory.create("openrouter", model="gpt-4")
        
        # Create from config dict
        config = {"provider": "gemini", "model": "gemini-pro"}
        client = ClientFactory.from_config(config)
    """
    
    # Mapping of provider names to client classes (lazy loaded)
    _providers: Dict[str, Type[BaseModelClient]] = None
    
    @classmethod
    def _ensure_providers_loaded(cls):
        """Lazy load provider mappings."""
        if cls._providers is None:
            # Import here to avoid circular imports
            from models.clients.openrouter_client import OpenRouterClient
            from models.clients.gemini_client import GeminiClient
            from models.clients.local_client import LocalModelClient
            
            cls._providers = {
                "openrouter": OpenRouterClient,
                "gemini": GeminiClient,
                "google": GeminiClient,
                "local": LocalModelClient,
                "huggingface": LocalModelClient,
                "transformers": LocalModelClient,
            }
            
            # Add any registered clients
            cls._providers.update(_CLIENT_REGISTRY)
    
    @classmethod
    def create(
        cls,
        provider: str,
        model: str = None,
        **kwargs
    ) -> BaseModelClient:
        """
        Create a model client for the specified provider.
        
        Args:
            provider: Provider name (openrouter, gemini, local)
            model: Model identifier
            **kwargs: Additional provider-specific arguments
        
        Returns:
            Configured BaseModelClient instance
        
        Raises:
            ValueError: If provider is not recognized
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
        
        # Build kwargs for client
        if model:
            kwargs["model"] = model
        
        return client_class(**kwargs)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseModelClient:
        """
        Create a client from a configuration dictionary.
        
        Args:
            config: Dict with 'provider', 'model', and optional parameters
        
        Returns:
            Configured BaseModelClient instance
        
        Example config:
            {
                "provider": "openrouter",
                "model": "x-ai/grok-4.1-fast:free",
                "base_url": "https://openrouter.ai/api/v1"
            }
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
        
        Supports two config formats:
        
        **New format (models section):**
            default_model: grok-fast
            models:
              grok-fast:
                provider: openrouter
                model: "x-ai/grok-4.1-fast:free"
              granite-code:
                provider: local
                model: "ibm-granite/granite-3b-code-instruct"
        
        **Legacy format (clients section):**
            clients:
              default: openrouter
              openrouter:
                model: "x-ai/grok-4.1-fast:free"
        
        Args:
            yaml_config: Full YAML config dict
            model_name: Name of model to create. Falls back to default_model / default.
        
        Returns:
            Configured BaseModelClient instance
        """
        # --- New format: 'models' section ---
        models_config = yaml_config.get("models")
        if models_config is not None:
            if model_name is None:
                model_name = yaml_config.get("default_model")
                if model_name is None:
                    # Pick the first model as fallback
                    model_name = next(iter(models_config))

            model_cfg = models_config.get(model_name)
            if model_cfg is None:
                available = ", ".join(models_config.keys())
                raise ValueError(
                    f"Model '{model_name}' not found in config. "
                    f"Available models: {available}"
                )
            model_cfg = model_cfg.copy()
            # 'provider' is required in the new format
            provider = model_cfg.pop("provider", model_name)
            return cls.from_config({"provider": provider, **model_cfg})

        # --- Legacy format: 'clients' section ---
        clients_config = yaml_config.get("clients", {})
        
        # Determine which client to use
        if model_name is None:
            model_name = clients_config.get("default", "openrouter")
        
        # Get client-specific config
        client_config = clients_config.get(model_name, {})
        
        # Add provider name if not in config
        if "provider" not in client_config:
            client_config["provider"] = model_name
        
        return cls.from_config(client_config)
    
    @classmethod
    def register_provider(cls, name: str, client_class: Type[BaseModelClient]):
        """
        Register a new provider/client class.
        
        Args:
            name: Provider name
            client_class: Client class that extends BaseModelClient
        
        Example:
            class MyCustomClient(BaseModelClient):
                ...
            
            ClientFactory.register_provider("my_provider", MyCustomClient)
        """
        cls._ensure_providers_loaded()
        
        if not issubclass(client_class, BaseModelClient):
            raise TypeError(
                f"Client class must extend BaseModelClient, "
                f"got {client_class.__name__}"
            )
        
        cls._providers[name.lower()] = client_class
        print(f"Registered provider: {name}")
    
    @classmethod
    def list_providers(cls) -> list:
        """Get list of available provider names."""
        cls._ensure_providers_loaded()
        return list(cls._providers.keys())
    
    @classmethod
    def list_models_from_config(cls, yaml_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        List all available models defined in a YAML config.
        
        Returns a dict of model_name -> {provider, model, ...} for each entry
        in the 'models' section, or synthesised entries from the legacy 'clients'
        section.
        """
        # New format
        models_config = yaml_config.get("models")
        if models_config is not None:
            return {
                name: cfg.copy()
                for name, cfg in models_config.items()
            }
        
        # Legacy format – build equivalent mapping
        clients_config = yaml_config.get("clients", {})
        result = {}
        for key, val in clients_config.items():
            if key == "default" or not isinstance(val, dict):
                continue
            entry = val.copy()
            entry.setdefault("provider", key)
            result[key] = entry
        return result

"""
Configuration Helpers

Functions for loading the YAML config and resolving model / API-key information.
"""

import os
import yaml


def get_config_path() -> str:
    """Get absolute path to the configuration file."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "config.yml")


def load_config() -> dict:
    """Load and return the YAML config as a dictionary."""
    config_path = get_config_path()
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def get_available_models() -> dict:
    """Return dict of model_name -> config from config.yml."""
    cfg = load_config()
    # New format
    models = cfg.get("models")
    if models:
        return models
    # Legacy fallback
    clients = cfg.get("clients", {})
    result = {}
    for k, v in clients.items():
        if k == "default" or not isinstance(v, dict):
            continue
        entry = v.copy()
        entry.setdefault("provider", k)
        result[k] = entry
    return result


def get_default_model_name() -> str:
    """Return the default model name from config."""
    cfg = load_config()
    name = cfg.get("default_model")
    if name:
        return name
    return cfg.get("clients", {}).get("default", "openrouter")


def validate_api_keys(model_choice) -> tuple:
    """Validate that necessary API keys are present for the selected model.
    
    Args:
        model_choice: model name string (from config) or legacy int.
    
    Returns:
        (is_valid: bool, message: str)
    """
    # Legacy integer support
    if isinstance(model_choice, int):
        legacy_map = {0: "local", 1: "openrouter", 2: "gemini"}
        provider = legacy_map.get(model_choice, "openrouter")
    else:
        models = get_available_models()
        model_cfg = models.get(model_choice, {})
        provider = model_cfg.get("provider", model_choice)

    provider_lower = provider.lower()
    if provider_lower == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            return False, (
                f"Missing OPENROUTER_API_KEY for model '{model_choice}'. "
                "Please add it to Streamlit Secrets or .env file."
            )
    elif provider_lower in ("gemini", "google"):
        if not os.getenv("GOOGLE_API_KEY"):
            return False, (
                f"Missing GOOGLE_API_KEY for model '{model_choice}'. "
                "Please add it to Streamlit Secrets or .env file."
            )
    # 'local' models don't need API keys

    return True, "OK"

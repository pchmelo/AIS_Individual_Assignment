import os
import yaml


def get_config_path() -> str:
    """Return the active config path.

    Priority:
    1. FAIRNESS_CONFIG_PATH env var (set by gui.launch() when a user config is supplied)
    2. Internal src/models/config.yml (default)
    """
    user_path = os.environ.get("FAIRNESS_CONFIG_PATH", "")
    if user_path and os.path.exists(user_path):
        return user_path
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
    return cfg.get("models", {})


def get_default_model_name() -> str:
    """Return the default model name from config."""
    cfg = load_config()
    return cfg.get("default_model", "")


def get_default_dataset() -> str:
    """Return the default dataset filename.

    Priority:
    1. FAIRNESS_DATASET_PATH env var (set by launch(dataset_path=...))
    2. ``dataset`` key in config.yml
    """
    env_dataset = os.environ.get("FAIRNESS_DATASET_PATH", "")
    if env_dataset and os.path.exists(env_dataset):
        return os.path.basename(env_dataset)
    cfg = load_config()
    return cfg.get("dataset", "")


def get_default_target_column() -> str:
    """Return the default target column from config (key: target_column)."""
    cfg = load_config()
    return cfg.get("target_column", "")


def validate_api_keys(model_choice: str) -> tuple:
    """Validate that necessary API keys are present for the selected model."""
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

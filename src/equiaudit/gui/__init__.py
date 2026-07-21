"""
GUI sub-package for the Fairness Evaluation System.

Requires additional dependencies (Streamlit). Install with:

    pip install -r requirements-gui.txt

Usage (from Python code):

    from gui import launch
    launch()

Or set ``mode: gui`` in your config.yml and run ``python main.py``.
"""


def _check_gui_deps() -> None:
    """Raise a friendly ImportError if Streamlit is not installed."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise ImportError(
            "\n"
            "GUI dependencies are not installed.\n"
            "\n"
            "Install them with:\n"
            "    pip install -r requirements-gui.txt\n"
            "\n"
            "Or install Streamlit directly:\n"
            "    pip install streamlit\n"
        ) from None


def launch(config_path: str = None, env_file: str = None, dataset_path: str = None) -> None:
    """Launch the Streamlit GUI application.

    Args:
        config_path: Path to a user config.yml. When provided the GUI will
            use this config for model selection and default settings.
            If omitted, the GUI uses its own internal config.
        env_file: Path to a .env file to load before launching. When omitted
            the function tries a .env next to ``config_path`` (if given),
            then falls back to a .env in the current working directory.
        dataset_path: Absolute (or relative) path to the CSV dataset the user
            wants to evaluate. When provided the GUI shows only this dataset
            and the bundled development datasets are hidden.

    Raises:
        ImportError: If Streamlit is not installed. Install with
            ``pip install -r requirements-gui.txt``.
    """
    _check_gui_deps()

    import os
    import subprocess
    import sys

    # ------------------------------------------------------------------
    # Load .env so API keys are present in the subprocess environment.
    # ------------------------------------------------------------------
    _dotenv_candidates = []
    if env_file:
        _dotenv_candidates.append(env_file)
    if config_path:
        _dotenv_candidates.append(os.path.join(os.path.dirname(os.path.abspath(config_path)), ".env"))
    _dotenv_candidates.append(os.path.join(os.getcwd(), ".env"))

    try:
        from dotenv import load_dotenv as _load_dotenv
        for _candidate in _dotenv_candidates:
            if os.path.exists(_candidate):
                _load_dotenv(_candidate, override=False)  # don't override already-set vars
                break
    except ImportError:
        pass  # python-dotenv not installed; rely on vars already in os.environ

    print("=" * 80)
    print("FAIRNESS EVALUATION - GUI Mode")
    print("=" * 80)
    print("Starting Streamlit application...")
    print("The web interface will open in your default browser.")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("=" * 80)

    gui_app_path = os.path.join(os.path.dirname(__file__), "app.py")
    src_dir = os.path.dirname(os.path.dirname(__file__))  # .../src/

    # Build env for subprocess — inherits current env (including any freshly
    # loaded .env vars) then adds PYTHONPATH and optionally FAIRNESS_CONFIG_PATH.
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir + (os.pathsep + existing if existing else "")

    if config_path:
        env["FAIRNESS_CONFIG_PATH"] = os.path.abspath(config_path)
    if dataset_path:
        env["FAIRNESS_DATASET_PATH"] = os.path.abspath(dataset_path)

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                gui_app_path,
                "--server.headless",
                "true",
            ],
            env=env,
        )
    except KeyboardInterrupt:
        print("\nShutting down GUI server...")

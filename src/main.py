from pipeline import DatasetEvaluationPipeline
import subprocess
import sys
import os

# ============================================
# MODE SELECTION: Change this variable to switch between terminal and GUI mode
# ============================================
RUN_MODE = "gui"  # Options: "terminal" or "gui"
# ============================================

# Client selection: "openrouter", "gemini", "local", or None to use config.yml
CLIENT_PROVIDER = None  # Set to None to use config.yml settings


def get_config_path():
    """Get path to the configuration file."""
    return os.path.join(os.path.dirname(__file__), "models", "config.yml")


def run_terminal_mode():
    print("Dataset Quality and Fairness Evaluation System")
    print("="*80)
    
    dataset_name = "adult-all"
    target_class = "Income"

    user_prompt = f"Evaluate the dataset '{dataset_name}' for data quality and fairness issues. Target: {target_class}. Provide a detailed report highlighting any problems found and suggestions for improvement."
    
    if not user_prompt:
        print("No input provided. Exiting.")
        return
    
    print(f"\nInitializing pipeline...")
    print(f"User prompt: {user_prompt}")
    
    try:
        if CLIENT_PROVIDER:
            # Legacy mode with explicit provider
            provider_map = {"openrouter": 1, "gemini": 2, "local": 0}
            use_api = provider_map.get(CLIENT_PROVIDER.lower(), 1)
            pipeline = DatasetEvaluationPipeline(use_api_model=use_api)
        else:
            # Use configuration file
            config_path = get_config_path()
            if os.path.exists(config_path):
                pipeline = DatasetEvaluationPipeline(config_path=config_path)
            else:
                print(f"Config file not found: {config_path}")
                print("Falling back to OpenRouter...")
                pipeline = DatasetEvaluationPipeline(use_api_model=1)
        
        results = pipeline.evaluate_dataset(user_prompt)
        pipeline.generate_report()
        print("\nEvaluation completed successfully.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

def run_gui_mode():
    print("Launching GUI mode...")
    print("="*80)
    print("Starting Streamlit application...")
    print("The web interface will open in your default browser.")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("="*80)
    
    try:
        gui_app_path = os.path.join(os.path.dirname(__file__), "gui", "app.py")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            gui_app_path,
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n\nShutting down GUI server...")
    except Exception as e:
        print(f"\nError launching GUI: {str(e)}")
        print("\nMake sure Streamlit is installed:")
        print("  pip install streamlit")

def main():
    if RUN_MODE.lower() == "gui":
        run_gui_mode()
    elif RUN_MODE.lower() == "terminal":
        run_terminal_mode()
    else:
        print(f"Error: Invalid RUN_MODE '{RUN_MODE}'")
        print("Please set RUN_MODE to either 'terminal' or 'gui' in main.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
     
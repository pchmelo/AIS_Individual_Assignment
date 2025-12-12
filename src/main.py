from pipeline import DatasetEvaluationPipeline
import subprocess
import sys
import os

# ============================================
# MODE SELECTION: Change this variable to switch between terminal and GUI mode
# ============================================
RUN_MODE = "gui"  # Options: "terminal" or "gui"
# ============================================

def run_terminal_mode():
    print("Dataset Quality and Fairness Evaluation System")
    print("="*80)
    
    # Model selection: 0=IBM Granite (Local), 1=Grok (API), 2=Gemini (API)
    use_api = 2
    datset_name = "adult-all"
    target_class = "Income"  # Specify target column for fairness analysis

    user_prompt = f"Evaluate the dataset '{datset_name}' for data quality and fairness issues. Target: {target_class}. Provide a detailed report highlighting any problems found and suggestions for improvement."
    
    if not user_prompt:
        print("No input provided. Exiting.")
        return
    
    model_names = {
        0: "IBM Granite (Local)",
        1: "Grok (API)",
        2: "Google Gemini (API)"
    }
    
    print(f"\nInitializing pipeline...")
    print(f"Model: {model_names.get(use_api, 'Unknown')}")
    print(f"User prompt: {user_prompt}")
    
    try:
        pipeline = DatasetEvaluationPipeline(use_api_model=use_api)
        results = pipeline.evaluate_dataset(user_prompt)
        pipeline.generate_report()
        print("\nEvaluation completed successfully.")
    except Exception as e:
        print(f"\nError: {str(e)}")

def run_gui_mode():
    print("Launching GUI mode...")
    print("="*80)
    print("Starting Streamlit application...")
    print("The web interface will open in your default browser.")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("="*80)
    
    try:
        gui_app_path = os.path.join(os.path.dirname(__file__), "gui_app.py")
        
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
     
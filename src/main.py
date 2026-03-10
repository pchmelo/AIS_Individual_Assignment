#!/usr/bin/env python
"""
Fairness Evaluation System - Main Entry Point

This application can run in three modes:
- GUI mode: Interactive Streamlit web interface
- CLI mode: Command-line interface with arguments
- Quick mode: Simple evaluation with hardcoded defaults

Usage:
    python main.py              # Runs in default mode (GUI)
    python main.py --gui        # Runs GUI mode
    python main.py --cli        # Runs CLI mode (pass additional args)
    python main.py --quick      # Quick evaluation with defaults
    
For CLI options:
    python -m cli --help
"""

import subprocess
import sys
import os

# ============================================
# DEFAULT MODE: Change this to set default behavior
# ============================================
DEFAULT_MODE = "gui"  # Options: "gui", "cli", "quick"
# ============================================


def get_config_path():
    """Get path to the configuration file."""
    return os.path.join(os.path.dirname(__file__), "models", "config.yml")


def run_quick_mode():
    """Run a quick evaluation with default settings."""
    print("=" * 80)
    print("FAIRNESS EVALUATION - Quick Mode")
    print("=" * 80)
    
    from cli import FairnessEvaluator
    
    # Default dataset and target
    dataset_name = "adult-all.csv"
    target_column = "Income"
    
    print(f"\nDataset: {dataset_name}")
    print(f"Target:  {target_column}")
    print()
    
    try:
        evaluator = FairnessEvaluator(
            config_path=get_config_path(),
            verbose=True,
        )
        
        # Verify first
        verify_result = evaluator.verify()
        if not verify_result.success:
            print("\nConfiguration verification failed!")
            for error in verify_result.errors:
                print(f"  ERROR: {error}")
            return 1
        
        # Run evaluation
        result = evaluator.evaluate(
            data=dataset_name,
            target=target_column,
        )
        
        if result.success:
            print("\n" + "=" * 80)
            print("EVALUATION COMPLETE")
            print("=" * 80)
            print(f"Report: {result.report_dir}")
            if result.pdf_path:
                print(f"PDF:    {result.pdf_path}")
            return 0
        else:
            print(f"\nEvaluation failed: {result.error}")
            return 1
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def run_cli_mode():
    """Run the full CLI with argument parsing."""
    print("Starting CLI mode...")
    from cli.__main__ import main as cli_main
    return cli_main()


def run_gui_mode():
    """Launch the Streamlit GUI."""
    print("=" * 80)
    print("FAIRNESS EVALUATION - GUI Mode")
    print("=" * 80)
    print("Starting Streamlit application...")
    print("The web interface will open in your default browser.")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("=" * 80)
    
    try:
        gui_app_path = os.path.join(os.path.dirname(__file__), "gui", "app.py")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            gui_app_path,
            "--server.headless", "true"
        ])
        return 0
    except KeyboardInterrupt:
        print("\n\nShutting down GUI server...")
        return 0
    except Exception as e:
        print(f"\nError launching GUI: {str(e)}")
        print("\nMake sure Streamlit is installed:")
        print("  pip install streamlit")
        return 1


def main():
    """Main entry point - dispatches to appropriate mode."""
    
    # Check for mode flags in arguments
    mode = DEFAULT_MODE
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("--gui", "-g"):
            mode = "gui"
            sys.argv.pop(1)  # Remove the flag
        elif arg in ("--cli", "-c"):
            mode = "cli"
            sys.argv.pop(1)
        elif arg in ("--quick", "-q"):
            mode = "quick"
            sys.argv.pop(1)
        elif arg in ("--help", "-h"):
            print(__doc__)
            print("\nFor CLI options, use: python -m cli --help")
            return 0
        elif not arg.startswith("-"):
            # If first arg is a file, assume CLI mode
            if arg.endswith(".csv") or os.path.exists(arg):
                mode = "cli"
                sys.argv.insert(1, "--data")  # Add --data flag before filename
    
    # Dispatch to appropriate mode
    if mode == "gui":
        return run_gui_mode()
    elif mode == "cli":
        return run_cli_mode()
    elif mode == "quick":
        return run_quick_mode()
    else:
        print(f"Error: Invalid mode '{mode}'")
        print("Valid modes: gui, cli, quick")
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
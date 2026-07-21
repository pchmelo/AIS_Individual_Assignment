#!/usr/bin/env python
"""
Fairness Evaluation System - Main Entry Point

This application can run in two modes:
- GUI mode: Interactive Streamlit web interface
- Quick mode: Headless evaluation driven by config.yml

Usage:
    python main.py              # Runs in default mode (GUI)
    python main.py --gui        # Runs GUI mode
    python main.py --quick      # Quick headless evaluation with defaults
    
For direct scripting, use example_usage.py instead.
"""

import sys
import os

# ============================================
# DEFAULT MODE: Change this to set default behavior.
# Can also be set via ``mode:`` in your config.yml.
# ============================================
DEFAULT_MODE = "gui"  # Options: "gui", "quick"
# ============================================


def _read_mode_from_config() -> str:
    """Read 'mode' from config.yml, falling back to DEFAULT_MODE.

    Search order:
    1. config.yml in the current working directory (user-facing config)
    2. src/models/config.yml (internal default)
    """
    try:
        import yaml
        candidates = [
            os.path.join(os.getcwd(), "config.yml"),   # e.g. examples/config.yml
            get_config_path(),                           # src/models/config.yml
        ]
        for config_path in candidates:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                mode = cfg.get("mode")
                if mode in ("gui", "quick"):
                    return mode
    except Exception:
        pass
    return DEFAULT_MODE


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


def run_gui_mode():
    """Launch the Streamlit GUI."""
    try:
        from gui import launch
        launch()
        return 0
    except ImportError as e:
        print(str(e))
        return 1
    except KeyboardInterrupt:
        return 0


def main():
    """Main entry point - dispatches to appropriate mode."""
    
    # Check for mode flags in arguments
    mode = _read_mode_from_config()
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("--gui", "-g"):
            mode = "gui"
            sys.argv.pop(1)  # Remove the flag
        elif arg in ("--quick", "-q"):
            mode = "quick"
            sys.argv.pop(1)
        elif arg in ("--help", "-h"):
            print(__doc__)
            return 0
        elif not arg.startswith("-"):
            # If first arg is a file, assume quick mode
            if arg.endswith(".csv") or os.path.exists(arg):
                mode = "quick"
                sys.argv.insert(1, "--data")  # Add --data flag before filename
    
    # Dispatch to appropriate mode
    if mode == "gui":
        return run_gui_mode()
    elif mode == "quick":
        return run_quick_mode()
    else:
        print(f"Error: Invalid mode '{mode}'")
        print("Valid modes: gui, quick")
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
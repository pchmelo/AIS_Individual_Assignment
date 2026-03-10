from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure src is in path
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def main(args: argparse.Namespace = None) -> int:
    """Main entry point for CLI."""
    
    if args is None:
        args = parse_args()
    
    # Handle environment variable loading from .env file
    if args.env_file:
        load_env_file(args.env_file)
    
    # Import after env loading to ensure API keys are available
    from cli.evaluator import FairnessEvaluator
    
    # Create evaluator
    evaluator = FairnessEvaluator(
        config_path=args.config,
        output_dir=args.output,
        model=args.model,
        verbose=not args.quiet,
    )
    
    # Verify mode
    if args.verify:
        result = evaluator.verify(test_prompt=args.test_api)
        return 0 if result.success else 1
    
    # Evaluate mode - requires data
    if not args.data:
        print("Error: --data is required for evaluation mode")
        print("Use --verify to check configuration only")
        return 1
    
    # Run evaluation
    result = evaluator.evaluate(
        data=args.data,
        target=args.target,
        objective=args.objective,
        sensitive_columns=args.sensitive.split(",") if args.sensitive else None,
        generate_pdf=not args.no_pdf,
    )
    
    if result.success:
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Dataset:    {result.dataset}")
        print(f"Target:     {result.target_column or 'auto-detected'}")
        print(f"Report Dir: {result.report_dir}")
        if result.pdf_path:
            print(f"PDF Report: {result.pdf_path}")
        if result.markdown_path:
            print(f"MD Report:  {result.markdown_path}")
        print(f"Stages:     {', '.join(result.stages_completed)}")
        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"  - {w}")
        return 0
    else:
        print("\n" + "=" * 60)
        print("EVALUATION FAILED")
        print("=" * 60)
        print(f"Error: {result.error}")
        return 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        prog="fairness-eval",
        description="Evaluate datasets for fairness and bias issues.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python -m cli --data adult-all.csv --target Income

  # Verify configuration
  python -m cli --verify --test-api

  # Custom config and output
  python -m cli --config my_config.yml --data data.csv --output ./reports

  # Specify model and sensitive columns
  python -m cli --data data.csv --model gemini-flash --sensitive "age,sex,race"

Environment Variables:
  OPENROUTER_API_KEY  API key for OpenRouter models
  GOOGLE_API_KEY      API key for Gemini models
        """,
    )
    
    # Data input
    parser.add_argument(
        "--data", "-d",
        type=str,
        help="Path to CSV dataset file (or filename if in src/data/)",
    )
    
    parser.add_argument(
        "--target", "-t",
        type=str,
        default=None,
        help="Target column name for classification (auto-detected if not specified)",
    )
    
    parser.add_argument(
        "--objective", "-o",
        type=str,
        default=None,
        help="Custom evaluation objective/prompt",
    )
    
    parser.add_argument(
        "--sensitive", "-s",
        type=str,
        default=None,
        help="Comma-separated list of sensitive columns (auto-detected if not specified)",
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: src/models/config.yml)",
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override default model from config",
    )
    
    parser.add_argument(
        "--env-file", "-e",
        type=str,
        default=None,
        help="Path to .env file for loading API keys",
    )
    
    # Output
    parser.add_argument(
        "--output", "-O",
        type=str,
        default=None,
        help="Output directory for reports (default: reports/)",
    )
    
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF generation",
    )
    
    # Modes
    parser.add_argument(
        "--verify", "-V",
        action="store_true",
        help="Verify configuration only (don't run evaluation)",
    )
    
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Test API connectivity during verification",
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages",
    )
    
    return parser.parse_args()


def load_env_file(env_path: str) -> None:
    """Load environment variables from a .env file."""
    env_path = Path(env_path)
    if not env_path.exists():
        print(f"Warning: .env file not found: {env_path}")
        return
    
    print(f"Loading environment from: {env_path}")
    
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value


if __name__ == "__main__":
    sys.exit(main())

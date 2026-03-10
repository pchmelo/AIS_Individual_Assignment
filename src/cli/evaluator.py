from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Add src to path for imports when running as standalone
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


@dataclass
class EvaluationResult:
    """Result of a fairness evaluation run."""
    
    success: bool
    dataset: str
    target_column: Optional[str]
    report_dir: str
    pdf_path: Optional[str]
    markdown_path: Optional[str]
    json_path: Optional[str]
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    stages_completed: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"EvaluationResult({status}, dataset={self.dataset}, report_dir={self.report_dir})"


@dataclass
class VerificationResult:
    """Result of configuration verification."""
    
    success: bool
    config_path: str
    model_name: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return f"VerificationResult({status}, model={self.model_name})"


@dataclass
class DoctorResult:
    """Result of the doctor() diagnostic check."""
    
    all_passed: bool
    checks: Dict[str, Dict[str, Any]]  # check_name -> {passed, message, details}
    
    def __repr__(self) -> str:
        passed = sum(1 for c in self.checks.values() if c.get("passed"))
        total = len(self.checks)
        return f"DoctorResult({passed}/{total} checks passed)"
    
    def summary(self) -> str:
        """Get a formatted summary of all checks."""
        lines = []
        for name, result in self.checks.items():
            status = "PASS" if result.get("passed") else "FAIL"
            lines.append(f"[{status}] {name}: {result.get('message', '')}")
            if result.get("details") and not result.get("passed"):
                lines.append(f"       {result.get('details')}")
        return "\n".join(lines)


class FairnessEvaluator:
    """
    Main class for running fairness evaluations on datasets.
    
    This class wraps the DatasetEvaluationPipeline and provides a simplified
    interface for headless (non-GUI) evaluation runs.
    
    Example:
        >>> evaluator = FairnessEvaluator(config_path="models/config.yml")
        >>> result = evaluator.verify()
        >>> if result.success:
        ...     eval_result = evaluator.evaluate("adult-all.csv", target="Income")
        ...     print(f"Report saved to: {eval_result.pdf_path}")
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize the FairnessEvaluator.
        
        Args:
            config_path: Path to the YAML configuration file. If None, uses default.
            output_dir: Base directory for reports. If None, uses project's reports/ folder.
            model: Override the default model from config. If None, uses config default.
            api_key: API key for the model provider (OpenRouter/Gemini). Overrides env variable.
            verbose: Whether to print progress messages.
        """
        self.verbose = verbose
        self.model_override = model
        self.api_key = api_key
        
        # Resolve config path
        if config_path is None:
            config_path = os.path.join(_SRC_DIR, "models", "config.yml")
        self.config_path = os.path.abspath(config_path)
        
        # Resolve output directory
        if output_dir is None:
            # Default to project root's reports/ folder
            project_root = os.path.dirname(_SRC_DIR)
            output_dir = os.path.join(project_root, "reports")
        self.output_dir = os.path.abspath(output_dir)
        
        # Pipeline will be initialized lazily
        self._pipeline = None
        self._initialized = False
        
        # Verbose initialization happens when evaluation starts
    
    def _ensure_initialized(self) -> None:
        """Lazily initialize the pipeline."""
        if self._initialized:
            return
        
        from pipeline.pipeline import DatasetEvaluationPipeline
        
        self._pipeline = DatasetEvaluationPipeline(
            config_path=self.config_path,
            default_model=self.model_override,
            api_key=self.api_key,
        )
        self._initialized = True
    
    def verify(self, test_prompt: bool = False) -> VerificationResult:
        """
        Verify that the configuration is valid and the system is ready.
        
        Args:
            test_prompt: If True, sends a test prompt to verify API connectivity.
        
        Returns:
            VerificationResult with status and any errors/warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []
        model_name = "unknown"
        
        # Check config file exists
        if not os.path.exists(self.config_path):
            errors.append(f"Configuration file not found: {self.config_path}")
            return VerificationResult(
                success=False,
                config_path=self.config_path,
                model_name=model_name,
                errors=errors,
                warnings=warnings,
            )
        
        # Try to initialize pipeline
        try:
            self._ensure_initialized()
            model_name = self._pipeline.agent_manager.config.get("default_model", "unknown")
            
            if self.verbose:
                print(f"Configuration loaded successfully")
                print(f"  Default model: {model_name}")
        except Exception as e:
            errors.append(f"Failed to initialize pipeline: {str(e)}")
            return VerificationResult(
                success=False,
                config_path=self.config_path,
                model_name=model_name,
                errors=errors,
                warnings=warnings,
            )
        
        # Check output directory
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.verbose:
                print(f"Output directory ready: {self.output_dir}")
        except Exception as e:
            errors.append(f"Cannot create output directory: {str(e)}")
        
        # Check API key if using OpenRouter
        model_config = self._pipeline.agent_manager.config.get("models", {}).get(model_name, {})
        provider = model_config.get("provider", "")
        
        if provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                warnings.append("OPENROUTER_API_KEY environment variable not set")
        elif provider == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                warnings.append("GOOGLE_API_KEY environment variable not set")
        
        # Optional: Test API connectivity
        if test_prompt and not errors:
            try:
                if self.verbose:
                    print("Testing API connectivity...")
                # Simple test message
                response = self._pipeline.model_client.chat.completions.create(
                    model=model_config.get("model", model_name),
                    messages=[{"role": "user", "content": "Say 'OK' if you can read this."}],
                    max_tokens=10,
                )
                if self.verbose:
                    print("  API test: OK")
            except Exception as e:
                errors.append(f"API connectivity test failed: {str(e)}")
        
        success = len(errors) == 0
        
        if self.verbose:
            if success:
                print("\nVerification: PASSED")
                if warnings:
                    print(f"  Warnings: {len(warnings)}")
                    for w in warnings:
                        print(f"    - {w}")
            else:
                print("\nVerification: FAILED")
                for e in errors:
                    print(f"  ERROR: {e}")
        
        return VerificationResult(
            success=success,
            config_path=self.config_path,
            model_name=model_name,
            errors=errors,
            warnings=warnings,
        )
    
    def doctor(self, dataset: Optional[str] = None) -> DoctorResult:
        """
        Run diagnostic checks to identify common problems before evaluation.
        
        Checks performed:
        - Configuration file exists and is valid YAML
        - Required dependencies are installed
        - API key is set and valid (makes a test request)
        - Model is available and responding
        - Dataset file exists (if provided)
        - Output directory is writable
        
        Args:
            dataset: Optional path to dataset to check. Can be a filename in data/
                     or a full path.
        
        Returns:
            DoctorResult with all check results.
        
        Example:
            >>> evaluator = FairnessEvaluator(api_key="sk-...")
            >>> result = evaluator.doctor("adult-all.csv")
            >>> if not result.all_passed:
            ...     print(result.summary())
        """
        checks: Dict[str, Dict[str, Any]] = {}
        
        def add_check(name: str, passed: bool, message: str, details: str = None):
            checks[name] = {"passed": passed, "message": message, "details": details}
            if self.verbose:
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {name}: {message}")
                if details and not passed:
                    print(f"         {details}")
        
        if self.verbose:
            print("\nRunning diagnostic checks...\n")
        
        # 1. Check config file
        if os.path.exists(self.config_path):
            try:
                import yaml
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                if config and isinstance(config, dict):
                    add_check("Config File", True, "Valid YAML configuration")
                else:
                    add_check("Config File", False, "Config file is empty or invalid")
            except Exception as e:
                add_check("Config File", False, "Failed to parse YAML", str(e))
        else:
            add_check("Config File", False, "Config file not found", self.config_path)
        
        # 2. Check dependencies
        missing_deps = []
        optional_missing = []
        for dep in ["pandas", "numpy", "yaml"]:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        for dep in ["reportlab", "aif360"]:
            try:
                __import__(dep)
            except ImportError:
                optional_missing.append(dep)
        
        if not missing_deps:
            msg = "All required packages installed"
            if optional_missing:
                msg += f" (optional missing: {', '.join(optional_missing)})"
            add_check("Dependencies", True, msg)
        else:
            add_check("Dependencies", False, "Missing required packages", 
                     f"Install: pip install {' '.join(missing_deps)}")
        
        # 3. Check model configuration
        try:
            self._ensure_initialized()
            model_name = self._pipeline.agent_manager.config.get("default_model", "unknown")
            model_config = self._pipeline.agent_manager.config.get("models", {}).get(model_name, {})
            provider = model_config.get("provider", "unknown")
            model_id = model_config.get("model", model_name)
            add_check("Model Config", True, f"{provider}/{model_id}")
        except Exception as e:
            add_check("Model Config", False, "Failed to load model config", str(e))
            # Can't continue without model config
            return DoctorResult(all_passed=False, checks=checks)
        
        # 4. Check API key
        api_key = self.api_key
        if not api_key:
            if provider == "openrouter":
                api_key = os.environ.get("OPENROUTER_API_KEY")
            elif provider in ["gemini", "google"]:
                api_key = os.environ.get("GOOGLE_API_KEY")
        
        if api_key:
            # Check key format
            if provider == "openrouter" and not api_key.startswith("sk-or-"):
                add_check("API Key Format", False, "Invalid OpenRouter key format",
                         "Key should start with 'sk-or-'. Get one at https://openrouter.ai/keys")
            elif provider in ["gemini", "google"] and len(api_key) < 20:
                add_check("API Key Format", False, "API key seems too short")
            else:
                add_check("API Key Format", True, "Key format looks valid")
        else:
            env_var = "OPENROUTER_API_KEY" if provider == "openrouter" else "GOOGLE_API_KEY"
            add_check("API Key Format", False, "No API key found",
                     f"Pass api_key parameter or set {env_var} environment variable")
        
        # 5. Test API connectivity (make a real request)
        if api_key and checks.get("API Key Format", {}).get("passed", False):
            try:
                test_messages = [{"role": "user", "content": "Reply with just: OK"}]
                response = self._pipeline.model_client.generate(test_messages, max_tokens=5)
                if response:
                    add_check("API Connection", True, "Model responding correctly")
                else:
                    add_check("API Connection", False, "Model returned empty response")
            except Exception as e:
                error_str = str(e)
                if "401" in error_str:
                    add_check("API Connection", False, "Authentication failed",
                             "Your API key is invalid or expired. Generate a new one.")
                elif "403" in error_str:
                    add_check("API Connection", False, "Access forbidden",
                             "Your key may not have access to this model.")
                elif "429" in error_str:
                    add_check("API Connection", False, "Rate limited",
                             "Too many requests. Wait a moment and try again.")
                elif "model" in error_str.lower() and "not found" in error_str.lower():
                    add_check("API Connection", False, "Model not found",
                             f"Model '{model_id}' is not available. Check the model name.")
                else:
                    add_check("API Connection", False, "Connection failed", error_str[:100])
        
        # 6. Check dataset (if provided)
        if dataset:
            dataset_path = None
            # Try different paths
            possible_paths = [
                dataset,
                os.path.join(_SRC_DIR, "data", dataset),
                os.path.join(_SRC_DIR, "data", f"{dataset}.csv") if not dataset.endswith('.csv') else None,
            ]
            possible_paths = [p for p in possible_paths if p]
            
            for path in possible_paths:
                if os.path.exists(path):
                    dataset_path = path
                    break
            
            if dataset_path:
                try:
                    df = pd.read_csv(dataset_path, nrows=5)
                    rows_info = "readable"
                    add_check("Dataset", True, f"Found at {os.path.basename(dataset_path)} ({len(df.columns)} columns)")
                except Exception as e:
                    add_check("Dataset", False, "File exists but cannot be read", str(e)[:80])
            else:
                add_check("Dataset", False, "Dataset not found",
                         f"Tried: {', '.join(os.path.basename(p) for p in possible_paths)}")
        
        # 7. Check output directory
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            test_file = os.path.join(self.output_dir, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            add_check("Output Directory", True, "Writable")
        except Exception as e:
            add_check("Output Directory", False, "Cannot write to output directory", str(e))
        
        all_passed = all(c.get("passed", False) for c in checks.values())
        
        if self.verbose:
            print()
            if all_passed:
                print("All checks passed! Ready to evaluate.")
            else:
                failed = sum(1 for c in checks.values() if not c.get("passed"))
                print(f"{failed} check(s) failed. Fix the issues above before evaluating.")
        
        return DoctorResult(all_passed=all_passed, checks=checks)

    def evaluate(
        self,
        data: Union[str, Path, pd.DataFrame],
        target: Optional[str] = None,
        objective: Optional[str] = None,
        sensitive_columns: Optional[List[str]] = None,
        ml_config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        generate_pdf: bool = True,
        max_pairs: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Run a fairness evaluation on a dataset.
        
        Args:
            data: Path to CSV file, or a pandas DataFrame.
            target: Target column name for classification. Auto-detected if None.
            objective: Custom objective/prompt for the evaluation.
            sensitive_columns: List of sensitive attribute columns. Auto-detected if None.
            ml_config: ML model configuration for fairness metrics.
            output_dir: Override output directory for this run.
            generate_pdf: Whether to generate PDF report.
            max_pairs: Maximum number of sensitive attribute pairs to analyze.
                      If set, the agent selects the most important pairs.
                      If None, uses value from config or analyzes all pairs.
        
        Returns:
            EvaluationResult with paths to generated reports.
        """
        self._ensure_initialized()
        
        warnings: List[str] = []
        stages_completed: List[str] = []
        
        # Resolve data input
        if isinstance(data, pd.DataFrame):
            # Save DataFrame to temporary CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_csv_name = f"temp_dataset_{timestamp}.csv"
            data_dir = os.path.join(_SRC_DIR, "data")
            os.makedirs(data_dir, exist_ok=True)
            temp_csv_path = os.path.join(data_dir, temp_csv_name)
            data.to_csv(temp_csv_path, index=False)
            dataset_name = temp_csv_name
            if self.verbose:
                print(f"DataFrame saved to: {temp_csv_path}")
        else:
            data_path = Path(data)
            if not data_path.exists():
                # Try looking in src/data/
                data_path = Path(_SRC_DIR) / "data" / data_path.name
            
            if not data_path.exists():
                return EvaluationResult(
                    success=False,
                    dataset=str(data),
                    target_column=target,
                    report_dir="",
                    pdf_path=None,
                    markdown_path=None,
                    json_path=None,
                    error=f"Dataset not found: {data}",
                )
            
            dataset_name = data_path.name
            
            # Copy to data/ if not already there
            expected_path = Path(_SRC_DIR) / "data" / dataset_name
            if data_path.resolve() != expected_path.resolve():
                os.makedirs(expected_path.parent, exist_ok=True)
                import shutil
                shutil.copy2(data_path, expected_path)
        
        # Build objective prompt
        if objective is None:
            target_str = f"Target: {target}. " if target else ""
            objective = (
                f"Evaluate the dataset '{dataset_name}' for data quality and fairness issues. "
                f"{target_str}Provide a detailed report highlighting any problems found and suggestions for improvement."
            )
        
        if self.verbose:
            print(f"\nEvaluating '{dataset_name}' (target: {target or 'auto-detect'})")
        
        try:
            # Load max_pairs from config if not provided
            if max_pairs is None:
                eval_config = self._pipeline.agent_manager.config.get("evaluation", {})
                max_pairs = eval_config.get("max_pairs")
            
            # Run the pipeline
            self._pipeline.evaluate_dataset(
                user_prompt=objective,
                confirmed_sensitive=sensitive_columns,
                ml_config=ml_config,
                max_pairs=max_pairs,
            )
            
            # Generate report
            self._pipeline.generate_report()
            
            report_dir = self._pipeline.report_dir
            md_path = os.path.join(report_dir, "evaluation_report.md")
            json_path = os.path.join(report_dir, "stage_data.json")
            
            # Determine PDF path
            pdf_path = None
            if generate_pdf:
                pdf_path = os.path.join(report_dir, "evaluation_report.pdf")
                if not os.path.exists(pdf_path):
                    pdf_path = None
                    warnings.append("PDF generation may have failed")
            
            # Get completed stages
            stages_completed = list(self._pipeline.evaluation_results.get("stages", {}).keys())
            
            if self.verbose:
                print(f"\nEvaluation completed successfully!")
                print(f"  Report directory: {report_dir}")
                print(f"  Stages completed: {len(stages_completed)}")
                if pdf_path:
                    print(f"  PDF report: {pdf_path}")
            
            return EvaluationResult(
                success=True,
                dataset=dataset_name,
                target_column=target or self._pipeline.target_column,
                report_dir=report_dir,
                pdf_path=pdf_path,
                markdown_path=md_path,
                json_path=json_path,
                warnings=warnings,
                stages_completed=stages_completed,
            )
        
        except Exception as e:
            from models.agents.base_agent import APIError
            
            # Handle API errors with clean messages
            if isinstance(e, APIError):
                error_msg = str(e)
                if self.verbose:
                    print(f"\nError: {error_msg}")
            else:
                error_msg = f"{type(e).__name__}: {str(e)}"
                if self.verbose:
                    import traceback
                    print(f"\nEvaluation failed: {error_msg}")
                    traceback.print_exc()
            
            return EvaluationResult(
                success=False,
                dataset=dataset_name,
                target_column=target,
                report_dir=getattr(self._pipeline, "report_dir", ""),
                pdf_path=None,
                markdown_path=None,
                json_path=None,
                error=error_msg,
                warnings=warnings,
                stages_completed=stages_completed,
            )


def create_evaluator(
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    model: Optional[str] = None,
    verbose: bool = True,
) -> FairnessEvaluator:
    """
    Factory function to create a FairnessEvaluator instance.
    
    This is a convenience function for creating evaluators with common configurations.
    
    Args:
        config_path: Path to YAML config file.
        output_dir: Base directory for reports.
        model: Override default model.
        verbose: Print progress messages.
    
    Returns:
        Configured FairnessEvaluator instance.
    """
    return FairnessEvaluator(
        config_path=config_path,
        output_dir=output_dir,
        model=model,
        verbose=verbose,
    )

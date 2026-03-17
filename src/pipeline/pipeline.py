from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from itertools import combinations as iter_combinations

# Root directory of the project (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agents.function_caller_agent import FunctionCallerAgent
from models.agents.data_analyst_agent import DataAnalystAgent
from models.agents.conversational_agent import ConversationalAgent
from models.agent_manager import AgentManager
from tools.fairness_tools import FairnessTools
from tools.bias_mitigation_tools import BiasMitigationTools

from pipeline.stage import Stage, NavigationAction
from pipeline.config import EVALUATION_STAGES, load_pipeline_config
from pipeline.stages.base import safe_json_dumps

from gui.pdf_generator import generate_pdf_bytes


class DatasetEvaluationPipeline:
    """
    Pipeline for evaluating datasets for quality and fairness issues.
    """
    def __init__(
        self,
        config_path: str = None,
        default_model: str = None,
        pipeline_config_path: str = None,
        api_key: str = None,
    ):
        self.fairness_tools = FairnessTools()
        self.bias_mitigation_tools = BiasMitigationTools()
        self.agent_manager: Optional[AgentManager] = None
        self.api_key = api_key

        self._stage_definitions = (
            load_pipeline_config(pipeline_config_path)
            if pipeline_config_path
            else EVALUATION_STAGES
        )

        # Determine config path
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "config.yml",
            )
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )
        
        self._init_from_config(config_path, default_model=default_model, api_key=api_key)

        self.current_dataset: Optional[str] = None
        self.user_objective: Optional[str] = None
        self.evaluation_results: Dict[str, Any] = {}

        # Dynamic pipeline state
        self._stages: List[Stage] = []
        self._current_stage_index: int = 0
        self._pipeline_ctx: Dict[str, Any] = {}

        # Pipeline ready


    def _init_from_config(self, config_path: str, default_model: str = None, api_key: str = None):
        self.agent_manager = AgentManager.from_yaml(config_path, api_key=api_key)
        if default_model:
            self.agent_manager.config["default_model"] = default_model
        self.model_client = self.agent_manager.get_client()
        self._initialize_agents()
        # Config loaded

    def _initialize_agents(self):
        self.file_parser_agent = self.agent_manager.get_primary_agent_for_stage("parsing")
        self.inspector_agent = self.agent_manager.get_primary_agent_for_stage("inspection")
        self.bias_mitigation_agent = self.agent_manager.get_primary_agent_for_stage("mitigation")
        self.quality_agent = self.agent_manager.get_primary_agent_for_stage("quality_analysis")
        self.fairness_agent = self.agent_manager.get_primary_agent_for_stage("fairness_analysis")
        self.recommendation_agent = self.agent_manager.get_primary_agent_for_stage("recommendation")

        if self.file_parser_agent is None:
            self.file_parser_agent = FunctionCallerAgent(
                tool_manager=self.fairness_tools, model_client=self.model_client,
                reflect_on_tool_use=True)
        if self.inspector_agent is None:
            self.inspector_agent = FunctionCallerAgent(
                tool_manager=self.fairness_tools, model_client=self.model_client,
                reflect_on_tool_use=True)
        if self.bias_mitigation_agent is None:
            self.bias_mitigation_agent = FunctionCallerAgent(
                tool_manager=self.bias_mitigation_tools, model_client=self.model_client,
                reflect_on_tool_use=True)
        if self.quality_agent is None:
            self.quality_agent = DataAnalystAgent(
                tool_manager=self.fairness_tools, model_client=self.model_client)
        if self.fairness_agent is None:
            self.fairness_agent = DataAnalystAgent(
                tool_manager=self.fairness_tools, model_client=self.model_client)
        if self.recommendation_agent is None:
            self.recommendation_agent = ConversationalAgent(model_client=self.model_client)

        # Agents ready


    def build_stages(
        self,
        dataset_name: str,
        target_column: Optional[str] = None,
        user_prompt: str = "",
    ) -> List[Stage]:
        """
        Build the ordered list of stages for an evaluation run.
        """
        self.current_dataset = dataset_name
        self.target_column = target_column
        self.user_objective = user_prompt

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = os.path.join(BASE_DIR, "reports", f"{dataset_name}_{timestamp}")
        self.images_dir = os.path.join(self.report_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        self.evaluation_results = {
            "dataset": dataset_name,
            "target_column": target_column,
            "user_objective": user_prompt,
            "report_directory": self.report_dir,
            "stages": {},
        }

        # Shared context dict that every stage can read/write
        self._pipeline_ctx = {
            "pipeline": self,
            "dataset_name": dataset_name,
            "target_column": target_column,
            "user_prompt": user_prompt,
            "report_dir": self.report_dir,
            "images_dir": self.images_dir,
            "confirmed_sensitive_columns": None,
            "ml_config": {"enabled": False},
            "selected_pairs": None,
            "mitigation_config": None,
            # Tool managers
            "fairness_tools": self.fairness_tools,
            "bias_mitigation_tools": self.bias_mitigation_tools,
            # Reference to the running results dict
            "results": self.evaluation_results["stages"],
        }

        stages: List[Stage] = []
        for defn in self._stage_definitions:
            if defn.requires_target and not target_column:
                continue
            agent = getattr(self, defn.agent_attr, None)
            stages.append(
                Stage(
                    key=defn.key,
                    name=defn.name,
                    execute_fn=defn.executor,
                    agent=agent,
                    description=defn.description,
                    optional=defn.optional,
                    requires_confirmation=defn.requires_confirmation,
                )
            )

        self._stages = stages
        self._current_stage_index = 0
        return stages


    @property
    def stages(self) -> List[Stage]:
        return self._stages

    @property
    def current_stage_index(self) -> int:
        return self._current_stage_index

    @current_stage_index.setter
    def current_stage_index(self, value: int):
        self._current_stage_index = max(0, min(value, len(self._stages) - 1))

    @property
    def current_stage(self) -> Optional[Stage]:
        if 0 <= self._current_stage_index < len(self._stages):
            return self._stages[self._current_stage_index]
        return None

    @property
    def pipeline_ctx(self) -> Dict[str, Any]:
        return self._pipeline_ctx

    @property
    def is_finished(self) -> bool:
        return self._current_stage_index >= len(self._stages)

    def navigate(self, action: NavigationAction, user_context: str = "") -> Dict[str, Any]:
        """Perform a navigation action and return the stage result."""
        if action == NavigationAction.BACKWARD:
            return self._go_backward(user_context)
        if action == NavigationAction.REPEAT:
            return self._go_repeat(user_context)
        return self._go_forward(user_context)

    def _go_forward(self, user_context: str = "") -> Dict[str, Any]:
        if self._current_stage_index >= len(self._stages):
            return {"status": "finished", "message": "All stages completed."}
        stage = self._stages[self._current_stage_index]
        stage.user_context = user_context or None
        result = stage.execute(self._pipeline_ctx)
        self.evaluation_results["stages"][stage.key] = result
        self._current_stage_index += 1
        
        # After sensitive attribute detection, handle pair selection (GUI mode)
        if stage.key == "3_sensitive":
            user_pairs = self._pipeline_ctx.get("user_specified_pairs")
            max_pairs = self._pipeline_ctx.get("max_pairs")
            # Only run if not already set (avoid overwriting user's chat-based overrides)
            if "selected_pairs" not in self._pipeline_ctx or self._pipeline_ctx["selected_pairs"] is None:
                self._handle_pair_selection(user_pairs, max_pairs)
        
        return result

    def _go_backward(self, user_context: str = "") -> Dict[str, Any]:
        if self._current_stage_index <= 0:
            return {"status": "info", "message": "Already at the first stage."}
        self._current_stage_index -= 1
        stage = self._stages[self._current_stage_index]
        stage.reset()
        self.evaluation_results["stages"].pop(stage.key, None)
        return {
            "status": "rewound",
            "message": f"Returned to **{stage.name}**. It will re-run on the next forward.",
        }

    def _go_repeat(self, user_context: str = "") -> Dict[str, Any]:
        idx = max(0, self._current_stage_index - 1)
        stage = self._stages[idx]
        stage.reset()
        stage.user_context = user_context or None
        self.evaluation_results["stages"].pop(stage.key, None)
        result = stage.execute(self._pipeline_ctx)
        self.evaluation_results["stages"][stage.key] = result
        return result

    # ==================================================================
    # Terminal-mode entry point
    # ==================================================================

    def evaluate_dataset(
        self,
        user_prompt: str,
        confirmed_sensitive: list = None,
        sensitive_pairs: list = None,
        ml_config: dict = None,
        max_pairs: int = None,
        mitigation_config: dict = None,
    ) -> Dict[str, Any]:
        """Run the full pipeline in one shot (used by terminal mode).
        
        Args:
            user_prompt: The evaluation objective/prompt.
            confirmed_sensitive: Pre-confirmed sensitive columns (skip detection).
            sensitive_pairs: Pre-defined pairs for intersectional analysis.
                            Each pair is a tuple/list of two column names.
                            If provided, uses these pairs directly (restricted mode).
                            If None, pairs are auto-selected based on max_pairs.
            ml_config: ML model configuration for fairness metrics.
            max_pairs: Maximum number of sensitive attribute pairs to analyze.
                      Only used when sensitive_pairs is None.
                      If set, the agent selects the most important pairs.
            mitigation_config: Bias mitigation configuration dict with format
                              {"methods": {"Reweighting": {}, "SMOTE": {}}}.
                              If None, the mitigation stage is skipped.
        """
        dataset_name = self._extract_dataset_name(user_prompt)
        target_column = self._extract_target_column(user_prompt)
        self.build_stages(dataset_name, target_column, user_prompt)

        if confirmed_sensitive:
            self._pipeline_ctx["confirmed_sensitive_columns"] = confirmed_sensitive
        if ml_config:
            self._pipeline_ctx["ml_config"] = ml_config
        if mitigation_config:
            self._pipeline_ctx["mitigation_config"] = mitigation_config
        
        # Store pair configuration in context
        self._pipeline_ctx["max_pairs"] = max_pairs
        self._pipeline_ctx["user_specified_pairs"] = sensitive_pairs

        # Count executable stages (skip mitigation if not configured)
        run_mitigation = bool(mitigation_config)
        executable_stages = [
            s for s in self._stages
            if s.key != "6_bias_mitigation" or run_mitigation
        ]
        total_stages = len(executable_stages)
        current_num = 0

        while not self.is_finished:
            stage = self.current_stage
            if stage.key == "6_bias_mitigation" and not run_mitigation:
                self._current_stage_index += 1
                continue
            current_num += 1
            print(f"[{current_num}/{total_stages}] {stage.name}")
            self.navigate(NavigationAction.FORWARD)
            
            # After sensitive attribute detection, handle pair selection
            if stage.key == "3_sensitive":
                self._handle_pair_selection(sensitive_pairs, max_pairs)

        print("Done.")

        return self.evaluation_results
    
    def _handle_pair_selection(self, sensitive_pairs: list = None, max_pairs: int = None) -> None:
        """
        Handle pair selection after sensitive attribute detection.
        
        Args:
            sensitive_pairs: User-specified pairs for restricted mode. If provided,
                           these pairs are used directly without agent selection.
            max_pairs: Maximum pairs for auto mode. If set, agent selects best pairs.
        """
        
        results = self.evaluation_results.get("stages", {})
        sensitive_cols = list(
            results.get("3_sensitive", {}).get("sensitive_columns", [])
        )
        
        # Exclude target column if present
        target = self._pipeline_ctx.get("target_column")
        if target and target in sensitive_cols:
            sensitive_cols = [c for c in sensitive_cols if c != target]
        
        if len(sensitive_cols) < 2:
            return  # No pairs possible
        
        all_pairs = list(iter_combinations(sensitive_cols, 2))
        
        if sensitive_pairs is not None:
            # Restricted mode: use user-specified pairs directly.
            # Pairs may reference any dataset column, not just the detected
            # sensitive columns. Validation against actual column existence
            # happens later in the fairness tool itself.
            valid_pairs = [(pair[0], pair[1]) for pair in sensitive_pairs if len(pair) == 2]
            
            if valid_pairs:
                self._pipeline_ctx["selected_pairs"] = valid_pairs
                self._pipeline_ctx["pair_selection_reasoning"] = (
                    f"User-specified pairs (restricted mode): {len(valid_pairs)} pair(s) selected."
                )
                self._save_pair_selection_to_results(
                    valid_pairs,
                    self._pipeline_ctx["pair_selection_reasoning"],
                    all_pairs,
                    len(valid_pairs),
                    mode="restricted"
                )
                print(f"  Using user-specified pairs: {[f'{p[0]}+{p[1]}' for p in valid_pairs]}")
            else:
                print("  Warning: No valid pairs from user specification, using all pairs")
                self._pipeline_ctx["selected_pairs"] = all_pairs
        
        elif max_pairs is not None:
            # Auto mode with limit: use agent to select best pairs
            self._select_best_pairs(max_pairs)
        
        else:
            # Auto mode without limit: use all pairs
            self._pipeline_ctx["selected_pairs"] = all_pairs
            self._pipeline_ctx["pair_selection_reasoning"] = (
                f"All {len(all_pairs)} pairs selected (no limit specified)."
            )
    
    def _select_best_pairs(self, max_pairs: int) -> None:
        """
        Use the agent to intelligently select the most important pairs
        for intersectional fairness analysis.
        """
        
        results = self.evaluation_results.get("stages", {})
        sensitive_cols = list(
            results.get("3_sensitive", {}).get("sensitive_columns", [])
        )
        
        # Exclude target column if present
        target = self._pipeline_ctx.get("target_column")
        if target and target in sensitive_cols:
            sensitive_cols = [c for c in sensitive_cols if c != target]
        
        if len(sensitive_cols) < 2:
            return  # No pairs possible
        
        all_pairs = list(iter_combinations(sensitive_cols, 2))
        
        if len(all_pairs) <= max_pairs:
            # No need to select, use all pairs
            self._pipeline_ctx["selected_pairs"] = all_pairs
            self._pipeline_ctx["pair_selection_reasoning"] = (
                f"All {len(all_pairs)} pairs selected (within max_pairs={max_pairs} limit)."
            )
            return
        
        # Ask agent to select best pairs
        print(f"  Selecting {max_pairs} most important pairs from {len(all_pairs)} possible...")
        
        pair_list_str = "\n".join([f"  - {p[0]} + {p[1]}" for p in all_pairs])
        
        prompt = f"""You are analyzing a dataset for fairness. The following sensitive attributes were detected:
{', '.join(sensitive_cols)}

All possible attribute pairs for intersectional analysis ({len(all_pairs)} total):
{pair_list_str}

Select exactly {max_pairs} pairs that are MOST IMPORTANT for fairness analysis. Consider:
1. Historical discrimination patterns (e.g., Race+Sex, Age+Gender are commonly studied)
2. Potential for intersectional bias (combinations that may compound disadvantage)
3. Relevance to employment/lending/healthcare fairness (depending on context)
4. Statistical significance (pairs that likely have enough data points)

Respond in this EXACT format:
SELECTED_PAIRS:
- Attribute1 + Attribute2
- Attribute3 + Attribute4

REASONING:
<Your explanation of why these pairs were selected>"""
        
        try:
            response = self.recommendation_agent.run(prompt)
            
            # Parse selected pairs from response
            selected = []
            reasoning = ""
            lines = response.strip().split("\n")
            in_pairs = False
            in_reasoning = False
            reasoning_lines = []
            
            for line in lines:
                line_stripped = line.strip()
                if "SELECTED_PAIRS:" in line.upper():
                    in_pairs = True
                    in_reasoning = False
                    continue
                if "REASONING:" in line.upper():
                    in_pairs = False
                    in_reasoning = True
                    continue
                
                if in_pairs and line_stripped.startswith("-"):
                    # Parse "- Attr1 + Attr2" format
                    pair_str = line_stripped[1:].strip()
                    if "+" in pair_str:
                        parts = [p.strip() for p in pair_str.split("+")]
                        if len(parts) == 2:
                            # Find matching pair (order-independent)
                            for p in all_pairs:
                                if (parts[0] == p[0] and parts[1] == p[1]) or \
                                   (parts[0] == p[1] and parts[1] == p[0]):
                                    if p not in selected:
                                        selected.append(p)
                                    break
                
                if in_reasoning:
                    reasoning_lines.append(line_stripped)
            
            reasoning = " ".join(reasoning_lines).strip()
            
            # Validate we got enough pairs
            if len(selected) < max_pairs:
                # Fall back to first N pairs if parsing failed
                print(f"  Warning: Only parsed {len(selected)} pairs, using first {max_pairs}")
                selected = all_pairs[:max_pairs]
                reasoning = f"Automatic selection: first {max_pairs} pairs (agent parsing issue)."
            elif len(selected) > max_pairs:
                selected = selected[:max_pairs]
            
            self._pipeline_ctx["selected_pairs"] = selected
            self._pipeline_ctx["pair_selection_reasoning"] = reasoning or "Agent selected these pairs based on fairness analysis criteria."
            
            # Save to evaluation results for the report
            self._save_pair_selection_to_results(selected, reasoning, all_pairs, max_pairs)
            
            print(f"  Selected pairs: {[f'{p[0]}+{p[1]}' for p in selected]}")
            
        except Exception as e:
            # Fallback: use first N pairs
            print(f"  Warning: Pair selection failed ({e}), using first {max_pairs} pairs")
            selected = all_pairs[:max_pairs]
            self._pipeline_ctx["selected_pairs"] = selected
            self._pipeline_ctx["pair_selection_reasoning"] = f"Automatic selection: first {max_pairs} pairs (fallback)."
            self._save_pair_selection_to_results(selected, self._pipeline_ctx["pair_selection_reasoning"], all_pairs, max_pairs)
    
    def _save_pair_selection_to_results(self, selected: list, reasoning: str, all_pairs: list, max_pairs: int, mode: str = "auto") -> None:
        """Save pair selection info to evaluation results for the report.
        
        Args:
            selected: List of selected pairs.
            reasoning: Explanation for why these pairs were selected.
            all_pairs: All possible pairs from detected sensitive columns.
            max_pairs: The max_pairs limit that was applied.
            mode: Selection mode - "auto" (agent-selected) or "restricted" (user-specified).
        """
        # Update the sensitive stage results with pair selection info
        if "3_sensitive" in self.evaluation_results.get("stages", {}):
            self.evaluation_results["stages"]["3_sensitive"]["pair_selection"] = {
                "mode": mode,
                "max_pairs_limit": max_pairs,
                "total_possible_pairs": len(all_pairs),
                "selected_pairs": [f"{p[0]} + {p[1]}" for p in selected],
                "reasoning": reasoning,
            }

        return self.evaluation_results

    # ---- prompt parsing helpers (terminal mode) ----------------------

    @staticmethod
    def _extract_dataset_name(user_prompt: str) -> str:
        prompt_lower = user_prompt.lower()
        words = user_prompt.split()

        for word in words:
            if ".csv" in word:
                return word.strip("'\"").replace(".csv", "")

        common = ["adult-all", "adult", "census", "credit", "compas", "german", "bank"]
        for ds in common:
            if ds in prompt_lower:
                return ds

        skip = {"audit", "analyze", "evaluate", "check", "inspect", "dataset", "the", "a", "an", "target"}
        remaining = [w.strip("'\"") for w in words if w.lower() not in skip and len(w) > 2]
        if remaining:
            return remaining[0].replace(".csv", "")
        return words[0].strip("'\"") if words else "dataset"

    @staticmethod
    def _extract_target_column(user_prompt: str) -> Optional[str]:
        prompt_lower = user_prompt.lower()
        if "target=" in prompt_lower or "target:" in prompt_lower:
            m = re.search(r"target[=:]\s*([a-zA-Z_-]+)", user_prompt, re.IGNORECASE)
            if m:
                return m.group(1)
        m = re.search(r"target\s+(?:is|as)\s+([a-zA-Z_-]+)", user_prompt, re.IGNORECASE)
        if m:
            return m.group(1)
        for t in ("income", "salary", "class", "label", "outcome", "result", "prediction"):
            if t in prompt_lower:
                return t
        return None

    # ==================================================================
    # Report generation
    # ==================================================================

    def generate_report(self, output_path: str = None) -> str:
        """Generate both markdown report and JSON data file."""        
        md_path = os.path.join(self.report_dir, "evaluation_report.md")
        json_path = os.path.join(self.report_dir, "stage_data.json")
        
        md_content = self._generate_markdown_report()
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Markdown report saved: {md_path}")
        
        json_content = self._generate_json_data()
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(safe_json_dumps(json_content))
        print(f"JSON data saved: {json_path}")
        
        self._save_fairness_comparison_files()
        
        # Generate and save PDF inside the report folder
        try:
            pdf_path = os.path.join(self.report_dir, "evaluation_report.pdf")
            pdf_bytes = generate_pdf_bytes(md_path)
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            print(f"PDF report saved: {pdf_path}")
        except Exception as e:
            print(f"Warning: Could not generate PDF: {e}")
        
        return md_content

    def _generate_markdown_report(self) -> str:
        """Generate pure markdown report (human-readable, easy PDF conversion)."""
        dataset_hash = hashlib.md5(self.current_dataset.encode()).hexdigest()[:8]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        lines: List[str] = []
        lines.append("# Dataset Fairness Evaluation Report")
        lines.append("")
        lines.append("## Metadata")
        lines.append("")
        lines.append(f"- **Dataset:** {self.current_dataset}")
        lines.append(f"- **Timestamp:** {ts}")
        lines.append(f"- **Dataset Hash:** {dataset_hash}")
        if hasattr(self, "target_column") and self.target_column:
            lines.append(f"- **Target Column:** {self.target_column}")
        lines.append(f"- **Objective:** {self.user_objective or 'Dataset auditing'}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        stage_titles = {
            "0_loading": "Stage 0: Dataset Loading",
            "1_objective": "Stage 1: Objective Inspection",
            "2_quality": "Stage 2: Data Quality Analysis",
            "3_sensitive": "Stage 3: Sensitive Attribute Detection",
            "4_imbalance": "Stage 4: Imbalance Analysis",
            "4_5_target_fairness": "Stage 4.5: Target Fairness Analysis",
            "5_recommendations": "Stage 5: Recommendations",
            "6_bias_mitigation": "Stage 6: Bias Mitigation",
        }
        
        for stage_key, stage_data in self.evaluation_results["stages"].items():
            title = stage_titles.get(stage_key, stage_key.replace("_", " ").title())
            lines.append(f"## {title}")
            lines.append("")
            
            if not isinstance(stage_data, dict):
                lines.append(str(stage_data))
                lines.append("")
                lines.append("---")
                lines.append("")
                continue
            
            if "tool_used" in stage_data:
                lines.append(f"**Tool Used:** `{stage_data['tool_used']}`")
                lines.append("")
            
            if stage_key == "6_bias_mitigation" and "methods" in stage_data:
                self._format_mitigation_markdown(lines, stage_data)
            elif "agent_analysis" in stage_data:
                lines.append("### Analysis")
                lines.append("")
                lines.append(stage_data["agent_analysis"])
                lines.append("")
                
                # Add pair selection info for sensitive stage
                if stage_key == "3_sensitive" and "pair_selection" in stage_data:
                    self._format_pair_selection_markdown(lines, stage_data["pair_selection"])
                    
            elif "recommendations" in stage_data:
                lines.append("### Recommendations")
                lines.append("")
                lines.append(stage_data["recommendations"])
                lines.append("")
            elif "agent_response" in stage_data:
                lines.append("### Response")
                lines.append("")
                lines.append(str(stage_data["agent_response"]))
                lines.append("")
            else:
                if "objective" in stage_data:
                    lines.append(f"**Objective:** {stage_data.get('objective', 'N/A')}")
                    lines.append("")
                if "validation" in stage_data:
                    lines.append(f"**Validation:** {stage_data.get('validation', 'N/A')}")
                    lines.append("")
            
            lines.append("---")
            lines.append("")
        
        lines.append("*Report generated by Dataset Fairness Evaluation System*")
        return "\n".join(lines)

    def _format_mitigation_markdown(self, lines: List[str], stage_data: Dict[str, Any]):
        """Format bias mitigation section as markdown."""
        methods_results = stage_data.get("methods", {})
        applied = stage_data.get("applied_methods", list(methods_results.keys()))
        
        lines.append(f"**Status:** {stage_data.get('status', 'unknown')}")
        lines.append(f"**Applied Methods:** {', '.join(applied)}")
        lines.append("")
        
        for method in applied:
            mr = methods_results.get(method, {})
            lines.append(f"### {method.replace('_', ' ').title()}")
            lines.append("")
            
            if mr.get("status") == "error":
                lines.append(f"**Error:** {mr.get('error', 'Unknown error')}")
                lines.append("")
                continue
            
            mitigation_result = mr.get("mitigation_result", {})
            if mitigation_result:
                lines.append("#### Mitigation Results")
                lines.append("")
                if "method" in mitigation_result:
                    lines.append(f"- **Technique:** {mitigation_result['method']}")
                if "original_rows" in mitigation_result and "new_rows" in mitigation_result:
                    orig = mitigation_result["original_rows"]
                    new = mitigation_result["new_rows"]
                    change = new - orig
                    pct = (change / orig * 100) if orig > 0 else 0
                    lines.append(f"- **Dataset Size:** {orig:,} → {new:,} ({pct:+.1f}%)")
                if "rows_added" in mitigation_result:
                    lines.append(f"- **Samples Added:** +{mitigation_result['rows_added']:,}")
                lines.append("")
            
            comparison_result = mr.get("comparison_result") or mitigation_result.get("comparison_result")
            if comparison_result:
                imb = comparison_result.get("imbalance_metrics", {})
                if imb:
                    lines.append("#### Imbalance Improvement")
                    lines.append("")
                    lines.append(f"- **Original Ratio:** {imb.get('original_imbalance_ratio', 'N/A'):.2f}")
                    lines.append(f"- **Mitigated Ratio:** {imb.get('mitigated_imbalance_ratio', 'N/A'):.2f}")
                    improvement = imb.get("improvement", "No")
                    lines.append(f"- **Improved:** {improvement}")
                    lines.append("")
                
                if "agent_analysis" in comparison_result:
                    lines.append("#### Agent Analysis")
                    lines.append("")
                    lines.append(comparison_result["agent_analysis"])
                    lines.append("")

    def _format_pair_selection_markdown(self, lines: List[str], pair_selection: Dict[str, Any]) -> None:
        """Format pair selection info as markdown."""
        lines.append("### Intersectional Pair Selection")
        lines.append("")
        lines.append(f"**Max Pairs Limit:** {pair_selection.get('max_pairs_limit', 'N/A')}")
        lines.append(f"**Total Possible Pairs:** {pair_selection.get('total_possible_pairs', 'N/A')}")
        lines.append("")
        lines.append("**Selected Pairs for Analysis:**")
        for pair in pair_selection.get("selected_pairs", []):
            lines.append(f"- {pair}")
        lines.append("")
        lines.append("**Selection Reasoning:**")
        lines.append("")
        lines.append(pair_selection.get("reasoning", "No reasoning provided."))
        lines.append("")

    def _generate_json_data(self) -> Dict[str, Any]:
        """Generate JSON file with all tool results organized by stage."""
        dataset_hash = hashlib.md5(self.current_dataset.encode()).hexdigest()[:8]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        json_data = {
            "metadata": {
                "dataset": self.current_dataset,
                "timestamp": ts,
                "dataset_hash": dataset_hash,
                "target_column": getattr(self, "target_column", None),
                "objective": self.user_objective or "Dataset auditing",
                "report_directory": self.report_dir,
            },
            "stages": {}
        }
        
        for stage_key, stage_data in self.evaluation_results["stages"].items():
            stage_json = {}
            
            if isinstance(stage_data, dict):
                if "tool_used" in stage_data:
                    stage_json["tool_used"] = stage_data["tool_used"]
                if "tool_result" in stage_data:
                    stage_json["tool_result"] = stage_data["tool_result"]
                if "pair_selection" in stage_data:
                    stage_json["pair_selection"] = stage_data["pair_selection"]
                if "ml_model_results" in stage_data:
                    stage_json["ml_model_results"] = stage_data["ml_model_results"]
                if "intersectional_ml_model_results" in stage_data:
                    stage_json["intersectional_ml_model_results"] = stage_data["intersectional_ml_model_results"]
                if "methods" in stage_data:
                    stage_json["methods"] = stage_data["methods"]
                    stage_json["applied_methods"] = stage_data.get("applied_methods", [])
                    stage_json["status"] = stage_data.get("status", "unknown")
            else:
                stage_json["data"] = stage_data
            
            json_data["stages"][stage_key] = stage_json
        
        return json_data

    def _save_fairness_comparison_files(self):
        """Save individual fairness comparison JSON files for each method."""
        stage_data = self.evaluation_results["stages"].get("6_bias_mitigation", {})
        methods_results = stage_data.get("methods", {})
        
        for method, mr in methods_results.items():
            mitigation_result = mr.get("mitigation_result", {})
            fairness_comparison = mr.get("fairness_comparison") or mitigation_result.get("fairness_comparison")
            
            if fairness_comparison and fairness_comparison.get("status") != "error":
                try:
                    fn = f"fairness_comparison_{method.lower().replace(' ', '_')}.json"
                    fp = os.path.join(self.report_dir, fn)
                    with open(fp, "w", encoding="utf-8") as f:
                        f.write(safe_json_dumps(fairness_comparison))
                except Exception as exc:
                    print(f"Warning: Could not save fairness comparison JSON: {exc}")



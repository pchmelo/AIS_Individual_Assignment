from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from models.agents.function_caller_agent import FunctionCallerAgent
from models.agents.data_analyst_agent import DataAnalystAgent
from models.agents.conversational_agent import ConversationalAgent
from models.agent_manager import AgentManager
from tools.fairness_tools import FairnessTools
from tools.bias_mitigation_tools import BiasMitigationTools

from pipeline.stage import Stage, NavigationAction
from pipeline.config import EVALUATION_STAGES, load_pipeline_config
from pipeline.stages.base import safe_json_dumps


class DatasetEvaluationPipeline:
    """Pipeline for evaluating datasets for quality and fairness issues.
    """
    def __init__(
        self,
        config_path: str = None,
        default_model: str = None,
        pipeline_config_path: str = None,
    ):
        self.fairness_tools = FairnessTools()
        self.bias_mitigation_tools = BiasMitigationTools()
        self.agent_manager: Optional[AgentManager] = None

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
        
        self._init_from_config(config_path, default_model=default_model)

        self.current_dataset: Optional[str] = None
        self.user_objective: Optional[str] = None
        self.evaluation_results: Dict[str, Any] = {}

        # Dynamic pipeline state
        self._stages: List[Stage] = []
        self._current_stage_index: int = 0
        self._pipeline_ctx: Dict[str, Any] = {}

        print("Pipeline initialized")

    # ---- initialisation helpers --------------------------------------

    def _init_from_config(self, config_path: str, default_model: str = None):
        self.agent_manager = AgentManager.from_yaml(config_path)
        if default_model:
            self.agent_manager.config["default_model"] = default_model
        self.model_client = self.agent_manager.get_client()
        self._initialize_agents()
        print(f"Loaded configuration from: {config_path}")

    def _initialize_agents(self):
        # Try to get each agent from config, fallback to defaults for missing ones
        self.file_parser_agent = self.agent_manager.get_primary_agent_for_stage("parsing")
        self.inspector_agent = self.agent_manager.get_primary_agent_for_stage("inspection")
        self.bias_mitigation_agent = self.agent_manager.get_primary_agent_for_stage("mitigation")
        self.quality_agent = self.agent_manager.get_primary_agent_for_stage("quality_analysis")
        self.fairness_agent = self.agent_manager.get_primary_agent_for_stage("fairness_analysis")
        self.recommendation_agent = self.agent_manager.get_primary_agent_for_stage("recommendation")

        # Fill in any missing agents with defaults
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

        print("All agents initialized from configuration")

    # ==================================================================
    # Stage-based pipeline API
    # ==================================================================

    def build_stages(
        self,
        dataset_name: str,
        target_column: Optional[str] = None,
        user_prompt: str = "",
    ) -> List[Stage]:
        """Build the ordered list of stages for an evaluation run.

        Uses the declarative :data:`EVALUATION_STAGES` config to produce
        :class:`Stage` instances with the correct executor and agent.
        """
        self.current_dataset = dataset_name
        self.target_column = target_column
        self.user_objective = user_prompt

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = os.path.join("reports", f"{dataset_name}_{timestamp}")
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
            "proxy_config": {"enabled": False},
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

    # ---- navigation --------------------------------------------------

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
    # Legacy terminal-mode entry point
    # ==================================================================

    def evaluate_dataset(
        self,
        user_prompt: str,
        confirmed_sensitive: list = None,
        proxy_config: dict = None,
    ) -> Dict[str, Any]:
        """Run the full pipeline in one shot (used by terminal mode)."""
        dataset_name = self._extract_dataset_name(user_prompt)
        target_column = self._extract_target_column(user_prompt)
        print(f"Evaluating dataset: {dataset_name}")
        if target_column:
            print(f"Target column detected: {target_column}")

        self.build_stages(dataset_name, target_column, user_prompt)

        if confirmed_sensitive:
            self._pipeline_ctx["confirmed_sensitive_columns"] = confirmed_sensitive
        if proxy_config:
            self._pipeline_ctx["proxy_config"] = proxy_config

        print(f"\n{'=' * 80}")
        print("DATASET EVALUATION PIPELINE")
        print(f"{'=' * 80}\n")

        while not self.is_finished:
            stage = self.current_stage
            if stage.key == "6_bias_mitigation":
                self._current_stage_index += 1
                continue
            print(f"\n{stage.name}")
            print("-" * 80)
            self.navigate(NavigationAction.FORWARD)

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED")
        print("=" * 80)

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
        if output_path is None:
            output_path = os.path.join(self.report_dir, "evaluation_report.txt")

        dataset_hash = hashlib.md5(self.current_dataset.encode()).hexdigest()[:8]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report: List[str] = []
        report.append("=" * 80)
        report.append("DATASET QUALITY AND FAIRNESS EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"\nDataset: {self.current_dataset}")
        report.append(f"Timestamp: {ts}")
        report.append(f"Dataset Hash: {dataset_hash}")
        report.append(f"Report Directory: {self.report_dir}")
        if hasattr(self, "target_column") and self.target_column:
            report.append(f"Target Column: {self.target_column}")
        report.append(f"User Objective: {self.user_objective or 'Dataset auditing'}")
        report.append("\n" + "=" * 80)

        stage_titles = {
            "0_loading": "STAGE 0: DATASET LOADING",
            "1_objective": "STAGE 1: OBJECTIVE INSPECTION",
            "2_quality": "STAGE 2: DATA QUALITY ANALYSIS",
            "3_sensitive": "STAGE 3: SENSITIVE ATTRIBUTE DETECTION",
            "4_imbalance": "STAGE 4: IMBALANCE ANALYSIS",
            "4_5_target_fairness": "STAGE 4.5: TARGET FAIRNESS ANALYSIS",
            "5_recommendations": "STAGE 5: RECOMMENDATIONS",
            "6_bias_mitigation": "STAGE 6: BIAS MITIGATION",
        }

        for stage_name, stage_data in self.evaluation_results["stages"].items():
            title = stage_titles.get(stage_name, stage_name.upper())
            report.append(f"\n\n{title}")
            report.append("-" * 80)

            if not isinstance(stage_data, dict):
                report.append(safe_json_dumps(stage_data))
                continue

            if stage_name == "6_bias_mitigation" and "methods" in stage_data:
                self._format_mitigation_report(report, stage_data)
            elif "tool_used" in stage_data:
                self._format_tool_stage_report(report, stage_data)
            elif "agent_analysis" in stage_data:
                report.append("\n\n[AGENT ANALYSIS]")
                report.append("-" * 80)
                report.append(stage_data["agent_analysis"])
            elif "recommendations" in stage_data:
                report.append("\n\n[RECOMMENDATIONS]")
                report.append("-" * 80)
                report.append(stage_data["recommendations"])
            else:
                report.append(safe_json_dumps(stage_data))

        report.append("\n\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        report_text = "\n".join(report)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Report saved: {output_path}")

        summary_path = os.path.join(self.report_dir, "agent_summary.txt")
        self._generate_agent_summary_report(summary_path)

        return report_text

    # ---- report formatting helpers -----------------------------------

    @staticmethod
    def _format_tool_stage_report(report: List[str], data: Dict[str, Any]):
        report.append(f"\n[TOOL USED]: {data['tool_used']}")
        report.append("")
        if "tool_result" in data:
            report.append("\n[TOOL RESULT]")
            report.append(safe_json_dumps(data["tool_result"]))
        if "proxy_model_results" in data:
            report.append("\n\n[PROXY MODEL RESULTS]")
            report.append(safe_json_dumps(data["proxy_model_results"]))
        if "intersectional_proxy_results" in data:
            report.append("\n\n[INTERSECTIONAL PROXY RESULTS]")
            report.append(safe_json_dumps(data["intersectional_proxy_results"]))
        if "agent_analysis" in data:
            report.append("\n\n[AGENT ANALYSIS]")
            report.append("-" * 80)
            report.append(data["agent_analysis"])
        elif "agent_response" in data:
            report.append("\n\n[AGENT RESPONSE]")
            report.append("-" * 80)
            report.append(str(data["agent_response"]))
        if "recommendations" in data:
            report.append("\n\n[RECOMMENDATIONS]")
            report.append("-" * 80)
            report.append(data["recommendations"])

    def _format_mitigation_report(self, report: List[str], stage_data: Dict[str, Any]):
        methods_results = stage_data["methods"]
        applied = stage_data.get("applied_methods", list(methods_results.keys()))

        report.append(f"\nStatus: {stage_data.get('status', 'unknown')}")
        report.append(f"Applied Methods: {', '.join(applied)}")
        report.append("")

        for method in applied:
            mr = methods_results.get(method, {})
            report.append(f"\n[{method.upper()}]")
            report.append("-" * 40)

            if mr.get("status") == "error":
                report.append(f"Error: {mr.get('error', 'Unknown error')}")
                continue

            mitigation_result = mr.get("mitigation_result", {})
            if mitigation_result:
                report.append("\n[MITIGATION RESULTS]")
                report.append(safe_json_dumps(mitigation_result))

            comparison_result = mr.get("comparison_result") or mitigation_result.get("comparison_result")
            if comparison_result:
                report.append("\n[COMPARISON RESULTS]")
                filtered = {k: v for k, v in comparison_result.items() if k != "agent_analysis"}
                report.append(safe_json_dumps(filtered))

            fairness_comparison = mr.get("fairness_comparison") or mitigation_result.get("fairness_comparison")
            if fairness_comparison and fairness_comparison.get("status") != "error":
                report.append("\n\n[FAIRNESS COMPARISON]")
                report.append(safe_json_dumps(fairness_comparison))
                try:
                    fn = f"fairness_comparison_{method.lower().replace(' ', '_')}.json"
                    fp = os.path.join(self.report_dir, fn)
                    with open(fp, "w", encoding="utf-8") as f:
                        f.write(safe_json_dumps(fairness_comparison))
                except Exception as exc:
                    print(f"Warning: Could not save fairness comparison JSON: {exc}")

            if comparison_result and "agent_analysis" in comparison_result:
                report.append("\n[AGENT ANALYSIS]")
                report.append(comparison_result["agent_analysis"])
            report.append("")

    def _generate_agent_summary_report(self, output_path: str):
        dataset_hash = hashlib.md5(self.current_dataset.encode()).hexdigest()[:8]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report: List[str] = []
        report.append("=" * 80)
        report.append("DATASET EVALUATION - AGENT SUMMARY")
        report.append("=" * 80)
        report.append(f"\nDataset: {self.current_dataset}")
        report.append(f"Timestamp: {ts}")
        report.append(f"Dataset Hash: {dataset_hash}")
        report.append(f"Report Directory: {self.report_dir}")
        if hasattr(self, "target_column") and self.target_column:
            report.append(f"Target Column: {self.target_column}")
        report.append(f"User Objective: {self.user_objective or 'Dataset auditing'}")
        report.append("\n" + "=" * 80)

        titles = {
            "0_loading": "STAGE 0: DATASET LOADING",
            "1_objective": "STAGE 1: OBJECTIVE INSPECTION",
            "2_quality": "STAGE 2: DATA QUALITY ANALYSIS",
            "3_sensitive": "STAGE 3: SENSITIVE ATTRIBUTE DETECTION",
            "4_imbalance": "STAGE 4: IMBALANCE ANALYSIS",
            "4_5_target_fairness": "STAGE 4.5: TARGET FAIRNESS ANALYSIS",
            "5_recommendations": "STAGE 5: RECOMMENDATIONS",
            "6_bias_mitigation": "STAGE 6: BIAS MITIGATION",
        }

        for stage_name, stage_data in self.evaluation_results["stages"].items():
            title = titles.get(stage_name, stage_name.upper())
            report.append(f"\n\n{title}")
            report.append("-" * 80)
            if isinstance(stage_data, dict):
                if "agent_analysis" in stage_data:
                    report.append(f"\n{stage_data['agent_analysis']}")
                elif "recommendations" in stage_data:
                    report.append(f"\n{stage_data['recommendations']}")
                else:
                    report.append(f"\n{safe_json_dumps(stage_data)}")
            else:
                report.append(f"\n{safe_json_dumps(stage_data)}")

        report.append("\n\n" + "=" * 80)
        report.append("END OF SUMMARY")
        report.append("=" * 80)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"Agent summary saved: {output_path}")



import json
import os
from typing import Dict, Any
from agents.function_caller_agent import FunctionCallerAgent
from agents.data_analyst_agent import DataAnalystAgent
from agents.conversational_agent import ConversationalAgent
from agents.model_client import LocalModelClient, OpenRouterClient
from tools.fairness_tools import FairnessTools

class DatasetEvaluationPipeline:
    def __init__(self, use_api_model: bool = False):
        self.fairness_tools = FairnessTools()
        
        if use_api_model:
            self.model_client = OpenRouterClient(
                model="x-ai/grok-4.1-fast:free",
            )
            print("Model: Grok (API)")
        else:
            self.model_client = LocalModelClient("ibm-granite/granite-3b-code-instruct")
            print("Model: IBM Granite (Local)")
        
        self._initialize_agents()
        
        self.current_dataset = None
        self.user_objective = None
        self.evaluation_results = {}
        
        print("Agents initialized")
    
    def _initialize_agents(self):
        self.file_parser_agent = FunctionCallerAgent(
            tool_manager=self.fairness_tools,
            model_client=self.model_client,
            reflect_on_tool_use=True
        )
        
        self.inspector_agent = FunctionCallerAgent(
            tool_manager=self.fairness_tools,
            model_client=self.model_client,
            reflect_on_tool_use=True
        )
        
        self.quality_agent = DataAnalystAgent(
            tool_manager=self.fairness_tools,
            model_client=self.model_client
        )
        
        self.fairness_agent = DataAnalystAgent(
            tool_manager=self.fairness_tools,
            model_client=self.model_client
        )
        
        self.recommendation_agent = ConversationalAgent(
            model_client=self.model_client
        )
        
        print("All agents initialized")
    
    def _extract_dataset_name(self, user_prompt: str) -> str:
        prompt_lower = user_prompt.lower()
        words = user_prompt.split()
        
        for word in words:
            if ".csv" in word:
                return word.replace(".csv", "")
        
        common_datasets = ["adult-all", "adult", "census", "credit", "compas", "german", "bank"]
        for dataset in common_datasets:
            if dataset in prompt_lower:
                return dataset
        
        action_words = ["audit", "analyze", "evaluate", "check", "inspect", "dataset", "the", "a", "an"]
        remaining_words = [w for w in words if w.lower() not in action_words and len(w) > 2]
        
        if remaining_words:
            return remaining_words[0].replace(".csv", "")
        
        return words[0] if words else "dataset"
    
    def evaluate_dataset(self, user_prompt: str) -> Dict[str, Any]:
        dataset_name = self._extract_dataset_name(user_prompt)
        print(f"Evaluating dataset: {dataset_name}")
        
        self.current_dataset = dataset_name

        self.user_objective = user_prompt
        self.evaluation_results = {
            "dataset": dataset_name,
            "user_objective": user_prompt,
            "stages": {}
        }
        
        print(f"\n{'='*80}")
        print(f"DATASET EVALUATION PIPELINE")
        print(f"{'='*80}\n")
        
        print("STAGE 0: Dataset Loading")
        print("-" * 80)
        
        dataset_load = self._stage_0_load_dataset(dataset_name)
        self.evaluation_results["stages"]["0_loading"] = dataset_load
        if dataset_load.get("status") != "success":
            print(f"Failed to load dataset: {dataset_load.get('message')}")
            return self.evaluation_results
        
        print("\nSTAGE 1: Objective Inspection")
        print("-" * 80)
        objective_check = self._stage_1_objective_inspection(user_prompt)
        self.evaluation_results["stages"]["1_objective"] = objective_check
        
        print("\nSTAGE 2: Dataset Overview")
        print("-" * 80)
        overview = self._stage_2_dataset_overview(dataset_name)
        self.evaluation_results["stages"]["2_overview"] = overview
        
        print("\nSTAGE 3: Data Quality Analysis")
        print("-" * 80)
        quality = self._stage_3_data_quality(dataset_name)
        self.evaluation_results["stages"]["3_quality"] = quality
        
        print("\nSTAGE 4: Sensitive Attribute Detection")
        print("-" * 80)
        sensitive = self._stage_4_sensitive_detection(dataset_name)
        self.evaluation_results["stages"]["4_sensitive"] = sensitive
        
        print("\nSTAGE 5: Imbalance Analysis")
        print("-" * 80)
        imbalance = self._stage_5_imbalance_analysis(dataset_name)
        self.evaluation_results["stages"]["5_imbalance"] = imbalance
        
        print("\nSTAGE 6: Findings Integration")
        print("-" * 80)
        integration = self._stage_6_integrate_findings()
        self.evaluation_results["stages"]["6_integration"] = integration
        
        print("\nSTAGE 7: Recommendations")
        print("-" * 80)
        recommendations = self._stage_7_recommendations()
        self.evaluation_results["stages"]["7_recommendations"] = recommendations
        
        print("\nSTAGE 8: Report Generation")
        print("-" * 80)
        self._stage_8_generate_report()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED")
        print("="*80)
        
        return self.evaluation_results
    
    def _stage_0_load_dataset(self, dataset_name: str) -> dict:
        prompt = f"Load the dataset named '{dataset_name}' using the load_dataset function."
        response = self.file_parser_agent.run(prompt)
        print(f"Dataset loading result: {response}")
        return response if isinstance(response, dict) else {"status": "success", "message": str(response)}
    
    def _stage_1_objective_inspection(self, user_objective: str) -> dict:
        is_audit_request = user_objective and any(kw in user_objective.lower() for kw in ["audit", "analyze", "evaluate", "check", "inspect"])
        
        result = {
            "objective": user_objective or "Dataset auditing",
            "is_audit_request": is_audit_request,
            "validation": "Dataset format compatible (CSV)"
        }
        print(f"Objective: {result['objective']}")
        print(f"Audit request: {result['is_audit_request']}")
        return result
    
    def _stage_2_dataset_overview(self, dataset_name: str) -> Dict[str, Any]:
        response = self.inspector_agent.run(f"Use analyze_fairness tool on {dataset_name}")
        print(f"Overview: {response}")
        return {"overview": response}
    
    def _stage_3_data_quality(self, dataset_name: str) -> Dict[str, Any]:
        print("Executing check_missing_data tool...")
        tool_result = self.fairness_tools.check_missing_data(dataset_name)
        print(f"Tool result: {json.dumps(tool_result, indent=2)}")
        
        print("\nAgent analyzing quality results...")
        analysis = self.quality_agent.run(f"Analyze this missing data report and provide insights: {json.dumps(tool_result)}")
        print(f"Quality analysis complete: {len(str(analysis))} chars")
        
        return {
            "tool_result": tool_result,
            "agent_analysis": analysis
        }
    
    def _create_simplified_column_summary(self, columns_data: list) -> str:
        """Create a simple table format for easier analysis"""
        summary = "COLUMN SUMMARY TABLE:\n"
        summary += "="*100 + "\n"
        summary += f"{'Column':<25} {'Type':<10} {'Unique':<8} {'Sample Values / Top Categories':<50}\n"
        summary += "="*100 + "\n"
        
        for col in columns_data:
            name = col['column']
            dtype = col['type']
            unique = col['unique_values']
            
            if 'top_values' in col:
                # Show top categories with percentages
                top_items = list(col['top_values'].items())[:3]
                values_str = ", ".join([f"{k}({v}%)" for k, v in top_items])
            else:
                # Show sample values for numeric
                values_str = str(col['sample_values'][:5])
            
            summary += f"{name:<25} {dtype:<10} {unique:<8} {values_str:<50}\n"
        
        return summary
    
    def _stage_4_sensitive_detection(self, dataset_name: str) -> Dict[str, Any]:
        print("Step 1: Executing detect_sensitive_attributes tool...")
        columns_result = self.fairness_tools.detect_sensitive_attributes(dataset_name)
        print(f"Tool result: {len(json.dumps(columns_result))} chars")
        
        # Create simplified summary
        simplified_summary = self._create_simplified_column_summary(columns_result.get('columns', []))
        print(f"\nSimplified summary:\n{simplified_summary}")
        
        print("\nStep 2: Agent identifying which columns are sensitive...")
        analysis_prompt = f"""Analyze this dataset and identify ALL SENSITIVE/PROTECTED attribute columns.

KEY SENSITIVE ATTRIBUTES TO LOOK FOR:
- Demographics: Age, Race, Ethnicity, Sex/Gender
- Personal: Religion, Marital-status, Relationship
- Socioeconomic: Income, Education, Occupation
- Geographic: Native-country, Nationality

{simplified_summary}

IMPORTANT: Look at BOTH column names AND their values/distributions:
- Race column with values like White, Black, Asian → SENSITIVE
- Sex column with Male/Female → SENSITIVE  
- Native-country with country names → SENSITIVE
- Age with numeric ages → SENSITIVE
- Education levels → SENSITIVE
- Marital-status → SENSITIVE
- Income/salary → SENSITIVE

For EACH sensitive column, output EXACTLY this format:
Column: [exact_column_name] | Reason: [why_sensitive] | Values: [key_values]

List ALL sensitive columns - don't miss Race, Sex, Native-country if present!"""
        
        # Use higher token limit for comprehensive analysis
        sensitive_list = self.recommendation_agent.run(analysis_prompt, max_tokens=1536)
        print(f"Sensitive columns identified ({len(sensitive_list)} chars)")
        print(f"Full response: {sensitive_list}")
        
        # Extract column names from the agent's response
        import re
        column_pattern = r'Column:\s*([\w-]+)'
        identified_columns = re.findall(column_pattern, sensitive_list)
        # Deduplicate while preserving order
        identified_columns = list(dict.fromkeys(identified_columns))
        print(f"Extracted sensitive column names: {identified_columns}")
        
        return {
            "tool_result": columns_result,
            "simplified_summary": simplified_summary,
            "agent_analysis": sensitive_list,
            "sensitive_columns": identified_columns
        }
    
    def _stage_5_imbalance_analysis(self, dataset_name: str) -> Dict[str, Any]:
        # Get sensitive columns from Stage 4
        sensitive_cols = self.evaluation_results["stages"]["4_sensitive"].get("sensitive_columns", [])
        print(f"Analyzing imbalance for {len(sensitive_cols)} sensitive columns: {sensitive_cols}")
        
        print("Executing check_class_imbalance tool...")
        tool_result = self.fairness_tools.check_class_imbalance(dataset_name)
        print(f"Tool result: {json.dumps(tool_result, indent=2)}")
        
        # Filter results to only sensitive columns
        if tool_result.get("status") == "success" and sensitive_cols:
            filtered_details = [
                detail for detail in tool_result.get("details", [])
                if detail["column"] in sensitive_cols
            ]
            tool_result["details"] = filtered_details
            tool_result["imbalanced_columns"] = len(filtered_details)
            print(f"Filtered to {len(filtered_details)} sensitive columns with imbalance")
        
        print("\nAgent analyzing imbalance in SENSITIVE columns only...")
        analysis_prompt = f"""Analyze class imbalance in SENSITIVE/PROTECTED attributes ONLY.

SENSITIVE COLUMNS IDENTIFIED: {', '.join(sensitive_cols)}

IMBALANCE DATA (for sensitive columns only):
{json.dumps(tool_result, indent=2)}

Provide:
1. Summary of imbalance severity for each sensitive column
2. Fairness risks (which groups are underrepresented?)
3. Impact on model bias
4. Specific mitigation recommendations

Focus ONLY on the sensitive columns listed above."""
        
        analysis = self.quality_agent.run(analysis_prompt)
        print(f"Imbalance analysis complete: {len(str(analysis))} chars")
        
        return {
            "tool_result": tool_result,
            "agent_analysis": analysis,
            "analyzed_columns": sensitive_cols
        }
    
    def _stage_6_integrate_findings(self) -> Dict[str, Any]:
        findings = self._compile_findings()
        summary = {
            "missing_issues": "See stage 3",
            "sensitive_risks": "See stage 4",
            "class_imbalance": "See stage 5",
            "severity": "Medium to High"
        }
        print(f"Integrated findings: {len(self.evaluation_results['stages'])} stages completed")
        return summary
    
    def _stage_7_recommendations(self) -> Dict[str, Any]:
        findings_summary = self._compile_findings()
        prompt = f"""Based on evaluation results for {self.current_dataset}, provide:
        1. Top 3 critical issues
        2. Mitigation strategies (SMOTE, reweighting, etc.)
        3. Priority order
        4. Expected impact
        
        Findings: {findings_summary}"""
        
        recommendations = self.recommendation_agent.run(prompt)
        print(f"Recommendations: {recommendations}")
        return {"recommendations": recommendations}
    
    def _stage_8_generate_report(self):
        self.generate_report()
        print("Report generated with timestamp and dataset hash")
    
    def _compile_findings(self) -> str:
        summary_parts = []
        
        for stage_name, stage_data in self.evaluation_results["stages"].items():
            summary_parts.append(f"\n{stage_name.upper()}:")
            summary_parts.append(json.dumps(stage_data, indent=2))
        
        return "\n".join(summary_parts)
    
    def generate_report(self, output_path: str = None):
        import hashlib
        
        if output_path is None:
            output_path = f"reports/{self.current_dataset.replace('.csv', '')}_evaluation_report.txt"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        dataset_hash = hashlib.md5(self.current_dataset.encode()).hexdigest()[:8]
        
        report = []
        report.append("=" * 80)
        report.append("DATASET QUALITY AND FAIRNESS EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"\nDataset: {self.current_dataset}")
        report.append(f"Timestamp: {self._get_timestamp()}")
        report.append(f"Dataset Hash: {dataset_hash}")
        report.append(f"User Objective: {self.user_objective or 'Dataset auditing'}")
        report.append("\n" + "=" * 80)
        
        stage_titles = {
            "0_loading": "STAGE 0: DATASET LOADING",
            "1_objective": "STAGE 1: OBJECTIVE INSPECTION",
            "2_overview": "STAGE 2: DATASET OVERVIEW",
            "3_quality": "STAGE 3: DATA QUALITY ANALYSIS",
            "4_sensitive": "STAGE 4: SENSITIVE ATTRIBUTE DETECTION",
            "5_imbalance": "STAGE 5: IMBALANCE ANALYSIS",
            "6_integration": "STAGE 6: FINDINGS INTEGRATION",
            "7_recommendations": "STAGE 7: RECOMMENDATIONS",
        }
        
        for stage_name, stage_data in self.evaluation_results["stages"].items():
            title = stage_titles.get(stage_name, stage_name.upper())
            report.append(f"\n\n{title}")
            report.append("-" * 80)
            
            # Format stage data with both JSON and readable text
            if isinstance(stage_data, dict):
                # Show JSON structure
                report.append("\n[STRUCTURED DATA]")
                report.append(json.dumps(stage_data, indent=2))
                
                # Show formatted agent analysis if present
                if "agent_analysis" in stage_data:
                    report.append("\n\n[AGENT ANALYSIS]")
                    report.append("-" * 80)
                    report.append(stage_data["agent_analysis"])
                
                # Show formatted recommendations
                if "recommendations" in stage_data:
                    report.append("\n\n[RECOMMENDATIONS]")
                    report.append("-" * 80)
                    report.append(stage_data["recommendations"])
            else:
                report.append(json.dumps(stage_data, indent=2))
        
        report.append("\n\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Report saved: {output_path}")
        
        return report_text
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.evaluation_results:
            return {"error": "No evaluation has been run yet"}
        
        return {
            "dataset": self.current_dataset,
            "stages_completed": len(self.evaluation_results.get("stages", {})),
            "has_recommendations": "recommendations" in self.evaluation_results.get("stages", {})
        }




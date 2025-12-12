import json
import os
import re
import hashlib
from datetime import datetime
from typing import Dict, Any
from agents.function_caller_agent import FunctionCallerAgent
from agents.data_analyst_agent import DataAnalystAgent
from agents.conversational_agent import ConversationalAgent
from agents.model_client import LocalModelClient, OpenRouterClient, GeminiClient
from tools.fairness_tools import FairnessTools


class DatasetEvaluationPipeline:
    def __init__(self, use_api_model: int = 0):
        self.fairness_tools = FairnessTools()
        
        if use_api_model == 1:
            self.model_client = OpenRouterClient(
                model="x-ai/grok-4.1-fast:free",
            )
            print("Model: Grok (API)")
        elif use_api_model == 2:
            self.model_client = GeminiClient(
                model="gemini-2.5-flash-lite"
            )
            print("Model: Google Gemini (API)")
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
                clean_word = word.strip("'\"").replace(".csv", "")
                return clean_word
        
        common_datasets = ["adult-all", "adult", "census", "credit", "compas", "german", "bank"]
        for dataset in common_datasets:
            if dataset in prompt_lower:
                return dataset
        
        action_words = ["audit", "analyze", "evaluate", "check", "inspect", "dataset", "the", "a", "an", "target"]
        remaining_words = [w.strip("'\"") for w in words if w.lower() not in action_words and len(w) > 2]
        
        if remaining_words:
            return remaining_words[0].replace(".csv", "")
        
        return words[0].strip("'\"") if words else "dataset"
    
    def _extract_target_column(self, user_prompt: str) -> str:
        prompt_lower = user_prompt.lower()
        
        # Look for "target=" or "target:" patterns
        if "target=" in prompt_lower or "target:" in prompt_lower:
            match = re.search(r'target[=:]\s*([a-zA-Z_-]+)', user_prompt, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Look for "target is/as" patterns
        match = re.search(r'target\s+(?:is|as)\s+([a-zA-Z_-]+)', user_prompt, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Look for common target keywords
        common_targets = ["income", "salary", "class", "label", "outcome", "result", "prediction"]
        for target in common_targets:
            if target in prompt_lower:
                return target
        
        return None
    
    def evaluate_dataset(self, user_prompt: str) -> Dict[str, Any]:
        dataset_name = self._extract_dataset_name(user_prompt)
        target_column = self._extract_target_column(user_prompt)
        print(f"Evaluating dataset: {dataset_name}")
        if target_column:
            print(f"Target column detected: {target_column}")
        
        self.current_dataset = dataset_name
        self.target_column = target_column
        self.user_objective = user_prompt
        
        # Create report directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = os.path.join("reports", f"{dataset_name}_{timestamp}")
        self.images_dir = os.path.join(self.report_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        print(f"Report directory: {self.report_dir}")
        
        self.evaluation_results = {
            "dataset": dataset_name,
            "target_column": target_column,
            "user_objective": user_prompt,
            "report_directory": self.report_dir,
            "stages": {}
        }
        
        print(f"\n{'='*80}")
        print(f"DATASET EVALUATION PIPELINE")
        print(f"{'='*80}\n")
        
        print("STAGE 0: Dataset Loading")
        print("-" * 80)
        
        dataset_load = self._stage_0_load_dataset(dataset_name)
        self.evaluation_results["stages"]["0_loading"] = dataset_load
        
        status = dataset_load.get("tool_result", {}).get("status") or dataset_load.get("status")
        if status != "success":
            message = dataset_load.get("tool_result", {}).get("message") or dataset_load.get("message", "Unknown error")
            print(f"Failed to load dataset: {message}")
            return self.evaluation_results
        
        print("\nSTAGE 1: Objective Inspection")
        print("-" * 80)
        objective_check = self._stage_1_objective_inspection(user_prompt)
        self.evaluation_results["stages"]["1_objective"] = objective_check
        
        print("\nSTAGE 2: Data Quality Analysis")
        quality = self._stage_2_data_quality(dataset_name)
        self.evaluation_results["stages"]["2_quality"] = quality
        
        print("\nSTAGE 3: Sensitive Attribute Detection")
        sensitive = self._stage_3_sensitive_detection(dataset_name)
        self.evaluation_results["stages"]["3_sensitive"] = sensitive
        
        print("\nSTAGE 4: Imbalance Analysis")
        imbalance = self._stage_4_imbalance_analysis(dataset_name)
        self.evaluation_results["stages"]["4_imbalance"] = imbalance
        
        # Optional Stage 4.5: Target Fairness Analysis (only if target is specified)
        if target_column:
            print("\nSTAGE 4.5: Target Fairness Analysis")
            target_fairness = self._stage_4_5_target_fairness_analysis(dataset_name, target_column)
            self.evaluation_results["stages"]["4_5_target_fairness"] = target_fairness
        
        print("\nSTAGE 5: Findings Integration")
        integration = self._stage_5_integrate_findings()
        self.evaluation_results["stages"]["5_integration"] = integration
        
        print("\nSTAGE 6: Recommendations")
        recommendations = self._stage_6_recommendations()
        self.evaluation_results["stages"]["6_recommendations"] = recommendations
        
        print("\nSTAGE 7: Report Generation")
        self._stage_7_generate_report()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED")
        print("="*80)
        
        return self.evaluation_results
    
    def _stage_0_load_dataset(self, dataset_name: str) -> dict:
        print("Tool: load_dataset")
        tool_result = self.fairness_tools.load_dataset(dataset_name)
        print(f"Tool result: {tool_result}")
        
        prompt = f"The dataset '{dataset_name}' has been loaded with the following information: {tool_result}. Provide a brief summary of the dataset."
        agent_analysis = self.file_parser_agent.run(prompt)
        print(f"Agent analysis: {agent_analysis}")
        
        return {
            "tool_used": "load_dataset",
            "tool_result": tool_result,
            "agent_analysis": agent_analysis
        }
    
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
    
    def _stage_2_data_quality(self, dataset_name: str) -> Dict[str, Any]:
        print("Tool: check_missing_data")
        tool_result = self.fairness_tools.check_missing_data(dataset_name)
        print(f"Tool result: {json.dumps(tool_result, indent=2)}")
        
        print("\nAgent analyzing quality results...")
        analysis = self.quality_agent.run(f"Analyze this missing data report and provide insights: {json.dumps(tool_result)}")
        print(f"Quality analysis complete: {len(str(analysis))} chars")
        
        return {
            "tool_used": "check_missing_data",
            "tool_result": tool_result,
            "agent_analysis": analysis
        }
    
    def _create_simplified_column_summary(self, columns_data: list) -> str:
        summary = "COLUMN SUMMARY TABLE:\n"
        summary += "="*100 + "\n"
        summary += f"{'Column':<25} {'Type':<10} {'Unique':<8} {'Sample Values / Top Categories':<50}\n"
        summary += "="*100 + "\n"
        
        for col in columns_data:
            name = col['column']
            dtype = col['type']
            unique = col['unique_values']
            
            if 'top_values' in col:
                top_items = list(col['top_values'].items())[:3]
                values_str = ", ".join([f"{k}({v}%)" for k, v in top_items])
            else:
                values_str = str(col['sample_values'][:5])
            
            summary += f"{name:<25} {dtype:<10} {unique:<8} {values_str:<50}\n"
        
        return summary
    
    def _stage_3_sensitive_detection(self, dataset_name: str) -> Dict[str, Any]:
        print("Tool: detect_sensitive_attributes")
        columns_result = self.fairness_tools.detect_sensitive_attributes(dataset_name)
        print(f"Tool result: {len(json.dumps(columns_result))} chars")
        
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

                                List ALL sensitive columns - don't miss Race, Sex, Native-country if present.
                            """
        
        sensitive_list = self.recommendation_agent.run(analysis_prompt, max_tokens=1536)
        print(f"Sensitive columns identified ({len(sensitive_list)} chars)")
        print(f"Full response: {sensitive_list}")
        
        column_pattern = r'Column:\s*([\w-]+)'
        identified_columns = re.findall(column_pattern, sensitive_list)
        
        identified_columns = list(dict.fromkeys(identified_columns))
        print(f"Extracted sensitive column names: {identified_columns}")
        
        return {
            "tool_used": "detect_sensitive_attributes",
            "tool_result": columns_result,
            "simplified_summary": simplified_summary,
            "agent_analysis": sensitive_list,
            "sensitive_columns": identified_columns
        }
    
    def _stage_4_imbalance_analysis(self, dataset_name: str) -> Dict[str, Any]:
        sensitive_cols = self.evaluation_results["stages"]["3_sensitive"].get("sensitive_columns", [])
        print(f"Analyzing imbalance for {len(sensitive_cols)} sensitive columns: {sensitive_cols}")
        
        print("Tool: check_class_imbalance")
        tool_result = self.fairness_tools.check_class_imbalance(dataset_name)
        print(f"Tool result: {json.dumps(tool_result, indent=2)}")
        
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

                                Focus ONLY on the sensitive columns listed above.
                            """
        
        analysis = self.quality_agent.run(analysis_prompt)
        print(f"Imbalance analysis complete: {len(str(analysis))} chars")
        
        return {
            "tool_used": "check_class_imbalance",
            "tool_result": tool_result,
            "agent_analysis": analysis,
            "analyzed_columns": sensitive_cols
        }
    
    def _stage_4_5_target_fairness_analysis(self, dataset_name: str, target_column: str, selected_pairs: list = None) -> Dict[str, Any]:
        """Analyze fairness metrics for target variable across sensitive attributes"""
        sensitive_cols = self.evaluation_results["stages"]["3_sensitive"].get("sensitive_columns", [])
        
        if not sensitive_cols:
            print("No sensitive columns identified, skipping target fairness analysis")
            return {
                "status": "skipped",
                "message": "No sensitive columns identified for fairness analysis"
            }
        
        print(f"Analyzing target '{target_column}' fairness across {len(sensitive_cols)} sensitive columns")
        print(f"Sensitive columns: {sensitive_cols}")
        
        # If selected_pairs is provided, only analyze those combinations
        if selected_pairs:
            print(f"User selected {len(selected_pairs)} combinations to analyze: {selected_pairs}")
        
        print("Tool: analyze_target_fairness")
        tool_result = self.fairness_tools.analyze_target_fairness(
            dataset_name=dataset_name,
            target_column=target_column,
            sensitive_columns=sensitive_cols,
            output_dir=self.images_dir,
            selected_pairs=selected_pairs  # Pass selected pairs to tool
        )
        
        if tool_result.get("status") == "success":
            print(f"Tool result: Generated {len(tool_result.get('generated_images', []))} visualizations")
            print(f"Images saved to: {self.images_dir}")
        else:
            print(f"Tool result: {tool_result}")
        
        print("\nAgent analyzing target fairness results...")
        analysis_prompt = f"""Analyze the target fairness metrics for '{target_column}' across sensitive attributes.

                                SENSITIVE COLUMNS ANALYZED: {', '.join(sensitive_cols)}

                                FAIRNESS METRICS DATA:
                                {json.dumps(tool_result, indent=2)}

                                Provide analysis on:
                                1. Target distribution across different demographic groups
                                2. Disparate impact - which groups have significantly different target rates?
                                3. Intersectional fairness - combined effects of multiple sensitive attributes
                                4. Statistical parity violations
                                5. Risk of discrimination or bias in predictions
                                6. Specific recommendations for achieving fairness

                                Focus on quantitative disparities and their implications.
                            """
        
        analysis = self.fairness_agent.run(analysis_prompt)
        print(f"Target fairness analysis complete: {len(str(analysis))} chars")
        
        return {
            "tool_used": "analyze_target_fairness",
            "tool_result": tool_result,
            "agent_analysis": analysis,
            "target_column": target_column,
            "analyzed_sensitive_columns": sensitive_cols
        }
    
    def _stage_5_integrate_findings(self) -> Dict[str, Any]:
        findings = self._compile_findings()
        summary = {
            "missing_issues": "See stage 2",
            "sensitive_risks": "See stage 3",
            "class_imbalance": "See stage 4",
            "severity": "Medium to High"
        }
        print(f"Integrated findings: {len(self.evaluation_results['stages'])} stages completed")
        return summary
    
    def _stage_6_recommendations(self) -> Dict[str, Any]:
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
    
    def _stage_7_generate_report(self):
        self.generate_report()
        print("Report generated with timestamp and dataset hash")
    
    def _compile_findings(self) -> str:
        summary_parts = []
        
        for stage_name, stage_data in self.evaluation_results["stages"].items():
            summary_parts.append(f"\n{stage_name.upper()}:")
            summary_parts.append(json.dumps(stage_data, indent=2))
        
        return "\n".join(summary_parts)
    
    def generate_report(self, output_path: str = None):
        if output_path is None:
            output_path = os.path.join(self.report_dir, "evaluation_report.txt")
        
        dataset_hash = hashlib.md5(self.current_dataset.encode()).hexdigest()[:8]
        
        report = []
        report.append("=" * 80)
        report.append("DATASET QUALITY AND FAIRNESS EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"\nDataset: {self.current_dataset}")
        report.append(f"Timestamp: {self._get_timestamp()}")
        report.append(f"Dataset Hash: {dataset_hash}")
        report.append(f"Report Directory: {self.report_dir}")
        if hasattr(self, 'target_column') and self.target_column:
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
            "5_integration": "STAGE 5: FINDINGS INTEGRATION",
            "6_recommendations": "STAGE 6: RECOMMENDATIONS",
        }
        
        for stage_name, stage_data in self.evaluation_results["stages"].items():
            title = stage_titles.get(stage_name, stage_name.upper())
            report.append(f"\n\n{title}")
            report.append("-" * 80)
            
            if isinstance(stage_data, dict):
                if "tool_used" in stage_data:
                    report.append(f"\n[TOOL USED]: {stage_data['tool_used']}")
                    report.append("")
                
                if "tool_result" in stage_data:
                    report.append("\n[TOOL RESULT]")
                    report.append(json.dumps(stage_data["tool_result"], indent=2))
                
                if "agent_analysis" in stage_data:
                    report.append("\n\n[AGENT ANALYSIS]")
                    report.append("-" * 80)
                    report.append(stage_data["agent_analysis"])
                
                if "agent_response" in stage_data and "agent_analysis" not in stage_data:
                    report.append("\n\n[AGENT RESPONSE]")
                    report.append("-" * 80)
                    report.append(str(stage_data["agent_response"]))
                
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
        
        simplified_output_path = os.path.join(self.report_dir, "agent_summary.txt")
        self._generate_agent_summary_report(simplified_output_path)
        
        return report_text
    
    def _generate_agent_summary_report(self, output_path: str):
        dataset_hash = hashlib.md5(self.current_dataset.encode()).hexdigest()[:8]
        
        report = []
        report.append("=" * 80)
        report.append("DATASET EVALUATION - AGENT SUMMARY")
        report.append("=" * 80)
        report.append(f"\nDataset: {self.current_dataset}")
        report.append(f"Timestamp: {self._get_timestamp()}")
        report.append(f"Dataset Hash: {dataset_hash}")
        report.append(f"Report Directory: {self.report_dir}")
        if hasattr(self, 'target_column') and self.target_column:
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
            "5_integration": "STAGE 5: FINDINGS INTEGRATION",
            "6_recommendations": "STAGE 6: RECOMMENDATIONS",
        }
        
        for stage_name, stage_data in self.evaluation_results["stages"].items():
            title = stage_titles.get(stage_name, stage_name.upper())
            report.append(f"\n\n{title}")
            report.append("-" * 80)
            
            if isinstance(stage_data, dict):
                if "agent_analysis" in stage_data:
                    report.append(f"\n{stage_data['agent_analysis']}")
                
                elif "agent_response" in stage_data:
                    report.append(f"\n{str(stage_data['agent_response'])}")
                
                elif "recommendations" in stage_data:
                    report.append(f"\n{stage_data['recommendations']}")
                
                else:
                    report.append(f"\n{json.dumps(stage_data, indent=2)}")
        
        report.append("\n\n" + "=" * 80)
        report.append("END OF SUMMARY")
        report.append("=" * 80)
        
        summary_text = "\n".join(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"Agent summary saved: {output_path}")
    
    def _get_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.evaluation_results:
            return {"error": "No evaluation has been run yet"}
        
        return {
            "dataset": self.current_dataset,
            "stages_completed": len(self.evaluation_results.get("stages", {})),
            "has_recommendations": "recommendations" in self.evaluation_results.get("stages", {})
        }




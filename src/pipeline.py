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
from tools.bias_mitigation_tools import BiasMitigationTools
import pandas as pd
import numpy as np


class DatasetEvaluationPipeline:
    def __init__(self, use_api_model: int = 0):
        self.fairness_tools = FairnessTools()
        self.bias_mitigation_tools = BiasMitigationTools()
        
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
        
        self.bias_mitigation_agent = FunctionCallerAgent(
            tool_manager=self.bias_mitigation_tools,
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
        
        if "target=" in prompt_lower or "target:" in prompt_lower:
            match = re.search(r'target[=:]\s*([a-zA-Z_-]+)', user_prompt, re.IGNORECASE)
            if match:
                return match.group(1)
        
        match = re.search(r'target\s+(?:is|as)\s+([a-zA-Z_-]+)', user_prompt, re.IGNORECASE)
        if match:
            return match.group(1)
        
        common_targets = ["income", "salary", "class", "label", "outcome", "result", "prediction"]
        for target in common_targets:
            if target in prompt_lower:
                return target
        
        return None
    
    
    def evaluate_dataset(self, user_prompt: str, confirmed_sensitive: list = None, 
                        proxy_config: dict = None) -> Dict[str, Any]:

        dataset_name = self._extract_dataset_name(user_prompt)
        target_column = self._extract_target_column(user_prompt)
        print(f"Evaluating dataset: {dataset_name}")
        if target_column:
            print(f"Target column detected: {target_column}")
        
        self.current_dataset = dataset_name
        self.target_column = target_column
        self.user_objective = user_prompt
        
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
        sensitive = self._stage_3_sensitive_detection(dataset_name, target_column)
        self.evaluation_results["stages"]["3_sensitive"] = sensitive
        
        
        if confirmed_sensitive:
             print(f"\nUsing confirmed sensitive columns: {confirmed_sensitive}")
             self.evaluation_results["stages"]["3_sensitive"]["sensitive_columns"] = confirmed_sensitive
        
        print("\nSTAGE 4: Imbalance Analysis")
        imbalance = self._stage_4_imbalance_analysis(dataset_name, proxy_config)
        self.evaluation_results["stages"]["4_imbalance"] = imbalance
        
        if target_column:
            print("\nSTAGE 4.5: Target Fairness Analysis")
            target_fairness = self._stage_4_5_target_fairness_analysis(dataset_name, target_column, proxy_config=proxy_config)
            self.evaluation_results["stages"]["4_5_target_fairness"] = target_fairness
        
        print("\nSTAGE 5: Recommendations")
        recommendations = self._stage_6_recommendations()
        self.evaluation_results["stages"]["5_recommendations"] = recommendations
        
        print("\nSTAGE 6: Report Generation")
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
    
    def _convert_to_serializable(self, obj):
        if isinstance(obj, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(element) for element in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_serializable(obj.tolist())
        else:
            return obj

    def _safe_json_dumps(self, data, indent=2):
        try:
            serializable_data = self._convert_to_serializable(data)
            return json.dumps(serializable_data, indent=indent)
        except Exception as e:
            return f"Error serializing data: {str(e)}"

    def _stage_2_data_quality(self, dataset_name: str) -> Dict[str, Any]:
        print("Tool: check_missing_data")
        tool_result = self.fairness_tools.check_missing_data(dataset_name)
        print(f"Tool result: {self._safe_json_dumps(tool_result)}")
        
        print("\nAgent analyzing quality results...")
        analysis = self.quality_agent.run(f"Analyze this missing data report and provide insights: {self._safe_json_dumps(tool_result)}")
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
    
    def _stage_3_sensitive_detection(self, dataset_name: str, target_column: str = None) -> Dict[str, Any]:
        print("Tool: detect_sensitive_attributes")
        columns_result = self.fairness_tools.detect_sensitive_attributes(dataset_name)
        print(f"Tool result: {len(self._safe_json_dumps(columns_result))} chars")
        
        simplified_summary = self._create_simplified_column_summary(columns_result.get('columns', []))
        print(f"\nSimplified summary:\n{simplified_summary}")
        
        print("\nStep 2: Agent identifying which columns are sensitive...")
        target_exclusion_note = ""
        if target_column:
            target_exclusion_note = f"\n\nIMPORTANT: EXCLUDE the target column '{target_column}' from sensitive attributes - it's the variable being predicted, not a protected attribute."
        
        analysis_prompt = f"""Analyze this dataset and identify ALL SENSITIVE/PROTECTED attribute columns.

                                KEY SENSITIVE ATTRIBUTES TO LOOK FOR:
                                - Demographics: Age, Race, Ethnicity, Sex/Gender
                                - Personal: Religion, Marital-status, Relationship
                                - Socioeconomic: Income, Education, Occupation
                                - Geographic: Native-country, Nationality

                                {simplified_summary}{target_exclusion_note}

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
        
        sensitive_list = self.recommendation_agent.run(analysis_prompt, max_tokens=4096)
        print(f"Sensitive columns identified ({len(sensitive_list)} chars)")
        print(f"Full response: {sensitive_list}")
        
        column_pattern = r'Column:\s*([\w-]+)'
        identified_columns = re.findall(column_pattern, sensitive_list)
        
        identified_columns = list(dict.fromkeys(identified_columns))
        
        if target_column and target_column in identified_columns:
            print(f"Removing target column '{target_column}' from sensitive attributes list")
            identified_columns.remove(target_column)
        
        print(f"Extracted sensitive column names (excluding target): {identified_columns}")
        
        return {
            "tool_used": "detect_sensitive_attributes",
            "tool_result": columns_result,
            "simplified_summary": simplified_summary,
            "agent_analysis": sensitive_list,
            "sensitive_columns": identified_columns
        }
    
    def _stage_4_imbalance_analysis(self, dataset_name: str, proxy_config: dict = None) -> Dict[str, Any]:
        sensitive_cols = self.evaluation_results["stages"]["3_sensitive"].get("sensitive_columns", [])
        print(f"Analyzing imbalance for {len(sensitive_cols)} sensitive columns: {sensitive_cols}")
        
        print("Tool: check_class_imbalance")
        tool_result = self.fairness_tools.check_class_imbalance(dataset_name)
        print(f"Tool result: {self._safe_json_dumps(tool_result)}")
        
        if tool_result.get("status") == "success" and sensitive_cols:
            filtered_details = [
                detail for detail in tool_result.get("details", [])
                if detail["column"] in sensitive_cols
            ]
            tool_result["details"] = filtered_details
            tool_result["imbalanced_columns"] = len(filtered_details)
            print(f"Filtered to {len(filtered_details)} sensitive columns with imbalance")
            
        proxy_results = None
        if proxy_config and proxy_config.get("enabled", False) and sensitive_cols and hasattr(self, 'target_column') and self.target_column:
            print("\nRunning Proxy Model Analysis (Single Columns)...")
            print(f"Config: {proxy_config}")
            proxy_results = self.fairness_tools.train_and_evaluate_proxy_model(
                dataset_name=dataset_name,
                target_column=self.target_column,
                sensitive_columns=sensitive_cols,
                test_size=proxy_config.get("test_size", 0.25),
                model_type=proxy_config.get("model_type", "Random Forest"),
                model_params=proxy_config.get("model_params", {})
            )
            print(f"Proxy Analysis Complete: {proxy_results.get('status')}")
        
        print("\nAgent analyzing imbalance in SENSITIVE columns only...")
        proxy_context = ""
        if proxy_results and proxy_results.get("status") == "success":
            per_label_str = ""
            if 'per_label_metrics' in proxy_results.get('performance', {}):
                per_label_str = "\nPer-Label Performance (F1, Precision, Recall):\n" + self._safe_json_dumps(proxy_results['performance']['per_label_metrics'])

            proxy_context = f"""
            PROXY MODEL FAIRNESS ANALYSIS:
            Model: {proxy_results.get('model_type')} (Acc: {proxy_results['performance']['accuracy']}, F1: {proxy_results['performance']['f1_macro']})
            
            {per_label_str}

            Fairness Metrics per Attribute (F1 Score & Disparity):
            {self._safe_json_dumps(proxy_results.get('fairness_analysis', {}))}
            
            Include these metrics (Statistical Parity, Disparate Impact, Group F1, FNR/FPR Ratios) in your assessment.
            
            CRITICAL ANALYSIS REQUIREMENTS:
            1. Compare "Base Rate" (Actual % Positive) vs "Selection Rate" (Predicted % Positive) for each group.
            2. High FNR (False Negative Rate) in a protected group means the model fails to select qualified candidates from that group. Highlight this.
            3. Calculate and mention the "FNR Ratio" (Max FNR / Min FNR) if significant disparity exists.
            4. Identify if the model *amplifies* existing bias (e.g. if Selection Rate disparity > Base Rate disparity).
            """

        analysis_prompt = f"""Analyze class imbalance in SENSITIVE/PROTECTED attributes ONLY.

                                SENSITIVE COLUMNS IDENTIFIED: {', '.join(sensitive_cols)}

                                IMBALANCE DATA (for sensitive columns only):
                                {self._safe_json_dumps(tool_result)}
                                {proxy_context}

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
            "proxy_model_results": proxy_results,
            "baseline_fairness_metrics": proxy_results, 
            "agent_analysis": analysis,
            "analyzed_columns": sensitive_cols
        }
    
    def _stage_4_5_target_fairness_analysis(self, dataset_name: str, target_column: str, selected_pairs: list = None,
                                          proxy_config: dict = None) -> Dict[str, Any]:
        sensitive_cols = self.evaluation_results["stages"]["3_sensitive"].get("sensitive_columns", [])
        
        if target_column in sensitive_cols:
            print(f"WARNING: Target column '{target_column}' found in sensitive columns, removing it")
            sensitive_cols = [col for col in sensitive_cols if col != target_column]
        
        if not sensitive_cols:
            print("No sensitive columns identified, skipping target fairness analysis")
            return {
                "status": "skipped",
                "message": "No sensitive columns identified for fairness analysis"
            }
        
        print(f"Analyzing target '{target_column}' fairness across {len(sensitive_cols)} sensitive columns")
        print(f"Sensitive columns: {sensitive_cols}")
        
        if selected_pairs:
            print(f"User selected {len(selected_pairs)} combinations to analyze: {selected_pairs}")
        
        print("Tool: analyze_target_fairness")
        tool_result = self.fairness_tools.analyze_target_fairness(
            dataset_name=dataset_name,
            target_column=target_column,
            sensitive_columns=sensitive_cols,
            output_dir=self.images_dir,
            selected_pairs=selected_pairs  
        )
        
        if tool_result.get("status") == "success":
            print(f"Tool result: Generated {len(tool_result.get('generated_images', []))} visualizations")
            print(f"Images saved to: {self.images_dir}")
        else:
            print(f"Tool result: {tool_result}")
            
        intersectional_proxy_results = None
        if proxy_config and proxy_config.get("enabled", False) and selected_pairs:
            try:
                path = self.fairness_tools._resolve_path(dataset_name)
                df = pd.read_csv(path)
                
                temp_cols = []
                for pair in selected_pairs:
                    col1, col2 = pair[0], pair[1]
                    if col1 in df.columns and col2 in df.columns:
                        combined_name = f"{col1}_{col2}_combined"
                        df[combined_name] = df[col1].astype(str) + "_" + df[col2].astype(str)
                        temp_cols.append(combined_name)
                
                if temp_cols:
                    temp_filename = f"temp_intersectional_{datetime.now().strftime('%H%M%S')}.csv"
                    temp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", temp_filename)
                    df.to_csv(temp_path, index=False)
                    
                    print(f"Running Proxy Model Intersectional Analysis on {temp_cols}...")
                    intersectional_proxy_results = self.fairness_tools.train_and_evaluate_proxy_model(
                        dataset_name=temp_filename.replace(".csv", ""),
                        target_column=target_column,
                        sensitive_columns=temp_cols,
                        test_size=proxy_config.get("test_size", 0.25),
                        model_type=proxy_config.get("model_type", "Random Forest"),
                        model_params=proxy_config.get("model_params", {})
                    )
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            except Exception as e:
                print(f"Intersectional proxy analysis failed: {e}")

        
        print("\nAgent analyzing target fairness results...")
        proxy_context = ""
        if intersectional_proxy_results and intersectional_proxy_results.get("status") == "success":
            proxy_context = f"""
            INTERSECTIONAL PROXY MODEL METRICS:
            (Model: {intersectional_proxy_results.get('model_type')})
            
            Fairness Metrics for Combined Groups (Intersectional):
            {self._safe_json_dumps(intersectional_proxy_results.get('fairness_analysis', {}))}
            
            CRITICAL ANALYSIS REQUIREMENTS:
            1. Analyze "F1 Score" for each intersectional group. Identify which SPECIFIC combination (e.g. Black Female) has the lowest performance.
            2. Compare "Base Rate" vs "Selection Rate".
            3. Highlight FNR Disparities. Are certain combinations being systematically rejected (High FNR)?
            """

        analysis_prompt = f"""Analyze the target fairness metrics for '{target_column}' across sensitive attributes.

                                SENSITIVE COLUMNS ANALYZED: {', '.join(sensitive_cols)}

                                FAIRNESS METRICS DATA:
                                {self._safe_json_dumps(tool_result)}
                                {proxy_context}

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
            "intersectional_proxy_results": intersectional_proxy_results,
            "agent_analysis": analysis,
            "target_column": target_column,
            "analyzed_sensitive_columns": sensitive_cols
        }
    
    def _stage_6_recommendations(self) -> Dict[str, Any]:
        print(f"Integrating findings from {len(self.evaluation_results['stages'])} stages completed")
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
            summary_parts.append(self._safe_json_dumps(stage_data))
        
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
            "6_bias_mitigation": "6_BIAS_MITIGATION",
        }
        
        for stage_name, stage_data in self.evaluation_results["stages"].items():
            title = stage_titles.get(stage_name, stage_name.upper())
            report.append(f"\n\n{title}")
            report.append("-" * 80)
            
            if isinstance(stage_data, dict):
                if stage_name == "6_bias_mitigation" and "methods" in stage_data:
                    methods_results = stage_data["methods"]
                    applied_methods = stage_data.get("applied_methods", list(methods_results.keys()))
                    
                    report.append(f"\nStatus: {stage_data.get('status', 'unknown')}")
                    report.append(f"Applied Methods: {', '.join(applied_methods)}")
                    report.append("")
                    
                    for method in applied_methods:
                        method_result = methods_results.get(method, {})
                        
                        report.append(f"\n[{method.upper()}]")
                        report.append("-" * 40)
                        
                        if method_result.get("status") == "error":
                            report.append(f"Error: {method_result.get('error', 'Unknown error')}")
                            continue
                        
                        mitigation_result = method_result.get("mitigation_result", {})
                        if mitigation_result:
                            report.append("\n[MITIGATION RESULTS]")
                            report.append(self._safe_json_dumps(mitigation_result))
                        
                        comparison_result = method_result.get("comparison_result")
                        if not comparison_result and mitigation_result:
                            comparison_result = mitigation_result.get("comparison_result")
                            
                        if comparison_result:
                            report.append("\n[COMPARISON RESULTS]")
                            comparison_without_analysis = {k: v for k, v in comparison_result.items() if k != "agent_analysis"}
                            report.append(self._safe_json_dumps(comparison_without_analysis))
                        
                        fairness_comparison = method_result.get("fairness_comparison")
                        if not fairness_comparison and mitigation_result:
                            fairness_comparison = mitigation_result.get("fairness_comparison")
                            
                        if fairness_comparison and fairness_comparison.get("status") != "error":
                            report.append("\n\n[FAIRNESS COMPARISON]")
                            report.append(self._safe_json_dumps(fairness_comparison))
                            
                            try:
                                fairness_json_filename = f"fairness_comparison_{method.lower().replace(' ', '_')}.json"
                                fairness_json_path = os.path.join(self.report_dir, fairness_json_filename)
                                with open(fairness_json_path, 'w', encoding='utf-8') as f:
                                    f.write(self._safe_json_dumps(fairness_comparison))
                                print(f"Saved fairness comparison JSON: {fairness_json_path}")
                            except Exception as e:
                                print(f"Warning: Could not save fairness comparison JSON: {e}")
                        
                        agent_analysis = comparison_result.get("agent_analysis")
                        if agent_analysis:
                            report.append("\n[AGENT ANALYSIS]")
                            report.append(agent_analysis)
                        
                        report.append("")
                
                elif "tool_used" in stage_data:
                    report.append(f"\n[TOOL USED]: {stage_data['tool_used']}")
                    report.append("")
                    
                    if "tool_result" in stage_data:
                        report.append("\n[TOOL RESULT]")
                        report.append(self._safe_json_dumps(stage_data["tool_result"]))
                    
                    if "proxy_model_results" in stage_data:
                        report.append("\n\n[PROXY MODEL RESULTS]")
                        report.append(self._safe_json_dumps(stage_data["proxy_model_results"]))
                    
                    if "intersectional_proxy_results" in stage_data:
                        report.append("\n\n[INTERSECTIONAL PROXY RESULTS]")
                        report.append(self._safe_json_dumps(stage_data["intersectional_proxy_results"]))
                    
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
                
                elif "agent_analysis" in stage_data:
                    report.append("\n\n[AGENT ANALYSIS]")
                    report.append("-" * 80)
                    report.append(stage_data["agent_analysis"])
                
                elif "agent_response" in stage_data:
                    report.append("\n\n[AGENT RESPONSE]")
                    report.append("-" * 80)
                    report.append(str(stage_data["agent_response"]))
                
                elif "recommendations" in stage_data:
                    report.append("\n\n[RECOMMENDATIONS]")
                    report.append("-" * 80)
                    report.append(stage_data["recommendations"])
                
                else:
                    report.append(self._safe_json_dumps(stage_data))
            else:
                report.append(self._safe_json_dumps(stage_data))
        
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
                    report.append(f"\n{self._safe_json_dumps(stage_data)}")
        
        report.append("\n\n" + "=" * 80)
        report.append("END OF SUMMARY")
        report.append("=" * 80)
        
        summary_text = "\n".join(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"Agent summary saved: {output_path}")
    
    def _get_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def apply_bias_mitigation(self, method: str, dataset_name: str, target_column: str,
                             sensitive_columns: list = None, **kwargs) -> Dict[str, Any]:
        try:
            output_dir = os.path.join(self.report_dir, "generated_csv")
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"\nApplying {method} bias mitigation...")
            
            if method == "reweighting":
                if not sensitive_columns:
                    return {"status": "error", "message": "Sensitive columns required for reweighting"}
                result = self.bias_mitigation_tools.apply_reweighting(
                    dataset_name=dataset_name,
                    target_column=target_column,
                    sensitive_columns=sensitive_columns,
                    output_dir=output_dir
                )
            elif method == "smote":
                k_neighbors = kwargs.get('k_neighbors', 5)
                sampling_strategy = kwargs.get('sampling_strategy', 'auto')
                result = self.bias_mitigation_tools.apply_smote(
                    dataset_name=dataset_name,
                    target_column=target_column,
                    output_dir=output_dir,
                    k_neighbors=k_neighbors,
                    sampling_strategy=sampling_strategy
                )
            elif method == "oversampling":
                sampling_strategy = kwargs.get('sampling_strategy', 'auto')
                result = self.bias_mitigation_tools.apply_oversampling(
                    dataset_name=dataset_name,
                    target_column=target_column,
                    output_dir=output_dir,
                    sampling_strategy=sampling_strategy
                )
            elif method == "undersampling":
                sampling_strategy = kwargs.get('sampling_strategy', 'auto')
                result = self.bias_mitigation_tools.apply_undersampling(
                    dataset_name=dataset_name,
                    target_column=target_column,
                    output_dir=output_dir,
                    sampling_strategy=sampling_strategy
                )
            else:
                return {"status": "error", "message": f"Unknown method: {method}"}
            
            if result.get("status") == "success" and result.get("output_file"):
                print(f"\n{'='*60}")
                print(f"FAIRNESS COMPARISON DEBUG for {method}")
                print(f"{'='*60}")
                
                baseline_metrics = self.evaluation_results.get("stages", {}).get("4_imbalance", {}).get("baseline_fairness_metrics")
                
                print(f"Baseline metrics found: {baseline_metrics is not None}")
                if baseline_metrics:
                    print(f"Baseline status: {baseline_metrics.get('status')}")
                
                if baseline_metrics and baseline_metrics.get("status") == "success":
                    output_file = result.get("output_file", "")
                    if output_file.endswith(".csv"):
                        csv_filename = output_file  
                    else:
                        csv_filename = output_file
                    
                    print(f"Mitigated CSV filename: {csv_filename}")
                    
                    proxy_config = kwargs.get("proxy_config", {
                        "test_size": 0.25,
                        "model_type": "Random Forest",
                        "model_params": {}
                    })
                    
                    analyzed_columns = self.evaluation_results.get("stages", {}).get("4_imbalance", {}).get("analyzed_columns", [])
                    
                    print(f"Analyzed columns: {analyzed_columns}")
                    print(f"Target column: {target_column}")
                    
                    if analyzed_columns and target_column:
                        print(f"\n✓ Running fairness comparison for {method}...")
                        
                        mitigated_metrics = self._run_proxy_on_mitigated_dataset(
                            csv_filename=csv_filename,
                            target_column=target_column,
                            sensitive_columns=analyzed_columns,
                            proxy_config=proxy_config
                        )
                        
                        print(f"Mitigated metrics status: {mitigated_metrics.get('status')}")
                        
                        if mitigated_metrics.get("status") == "success":
                            comparison = self._compare_fairness_metrics(
                                baseline=baseline_metrics,
                                mitigated=mitigated_metrics,
                                method_name=method
                            )
                            
                            result["fairness_comparison"] = comparison
                            print(f"Fairness comparison complete: {comparison.get('overall_improvement', 'Unknown')} improvement")
                            print(f"{'='*60}\n")
                        else:
                            print(f"Could not run proxy model on mitigated dataset: {mitigated_metrics.get('message', 'Unknown error')}")
                            print(f"{'='*60}\n")
                    else:
                        print(f"Skipping fairness comparison: missing sensitive columns or target column")
                        print(f"{'='*60}\n")
                else:
                    print(f"Skipping fairness comparison: no baseline metrics available from Stage 4")
                    print(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def compare_mitigation_results(self, original_dataset: str, mitigated_dataset: str,
                                   target_column: str, sensitive_columns: list) -> Dict[str, Any]:
        try:
            print("\nComparing original and mitigated datasets...")
            result = self.bias_mitigation_tools.compare_datasets(
                original_dataset=original_dataset,
                mitigated_dataset=mitigated_dataset,
                target_column=target_column,
                sensitive_columns=sensitive_columns
            )
            
            analysis_prompt = f"""Analyze the comparison between original and mitigated datasets:
            
            {self._safe_json_dumps(result)}
            
            Provide a detailed analysis:
            1. Was the bias mitigation effective? (Yes/No and why)
            2. What improved? (specific metrics and percentages)
            3. What remained problematic? (if any)
            4. Recommendations for further improvements
            
            Be specific with numbers and provide actionable insights."""
            
            agent_analysis = self.recommendation_agent.run(analysis_prompt)
            
            result["agent_analysis"] = agent_analysis
            
            return result
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _run_proxy_on_mitigated_dataset(self, csv_filename: str, target_column: str,
                                        sensitive_columns: list, proxy_config: dict = None) -> Dict[str, Any]:
        try:
            if proxy_config is None:
                proxy_config = {
                    "test_size": 0.25,
                    "model_type": "Random Forest",
                    "model_params": {}
                }
            
            print(f"\nRunning proxy model on mitigated dataset: {csv_filename}")
            proxy_results = self.fairness_tools.train_and_evaluate_proxy_model(
                dataset_name=csv_filename,
                target_column=target_column,
                sensitive_columns=sensitive_columns,
                test_size=proxy_config.get("test_size", 0.25),
                model_type=proxy_config.get("model_type", "Random Forest"),
                model_params=proxy_config.get("model_params", {})
            )
            
            return proxy_results
            
        except Exception as e:
            print(f"Error running proxy model on mitigated dataset: {e}")
            return {"status": "error", "message": str(e)}
    
    def _compare_fairness_metrics(self, baseline: dict, mitigated: dict, method_name: str) -> Dict[str, Any]:
        try:
            if not baseline or baseline.get("status") != "success":
                return {"status": "error", "message": "Invalid baseline metrics"}
            
            if not mitigated or mitigated.get("status") != "success":
                return {"status": "error", "message": "Invalid mitigated metrics"}
            
            comparison = {
                "method": method_name,
                "baseline_metrics": baseline,
                "mitigated_metrics": mitigated,
                "improvements": {},
                "per_attribute_comparison": {}
            }
            
            baseline_fairness = baseline.get("fairness_analysis", {})
            mitigated_fairness = mitigated.get("fairness_analysis", {})
            
            for attr in baseline_fairness.keys():
                if attr not in mitigated_fairness:
                    continue
                
                baseline_metrics = baseline_fairness[attr].get("metrics", {})
                mitigated_metrics = mitigated_fairness[attr].get("metrics", {})
                
                spd_baseline = baseline_metrics.get("statistical_parity_difference", 0)
                spd_mitigated = mitigated_metrics.get("statistical_parity_difference", 0)
                spd_improvement = spd_baseline - spd_mitigated 
                
                di_baseline = baseline_metrics.get("disparate_impact", 0)
                di_mitigated = mitigated_metrics.get("disparate_impact", 0)
                di_improvement = di_mitigated - di_baseline  
                
                comparison["per_attribute_comparison"][attr] = {
                    "statistical_parity_difference": {
                        "baseline": float(spd_baseline), 
                        "mitigated": float(spd_mitigated),
                        "change": float(spd_improvement),
                        "improved": bool(abs(spd_mitigated) < abs(spd_baseline)) 
                    },
                    "disparate_impact": {
                        "baseline": float(di_baseline),
                        "mitigated": float(di_mitigated),
                        "change": float(di_improvement),
                        "improved": bool(abs(1.0 - di_mitigated) < abs(1.0 - di_baseline)) 
                    }
                }
            
            improvements_count = sum(
                1 for attr_comp in comparison["per_attribute_comparison"].values()
                if attr_comp["statistical_parity_difference"]["improved"] or 
                   attr_comp["disparate_impact"]["improved"]
            )
            total_metrics = len(comparison["per_attribute_comparison"]) * 2
            
            if improvements_count > total_metrics * 0.6:
                comparison["overall_improvement"] = "Significant"
            elif improvements_count > total_metrics * 0.3:
                comparison["overall_improvement"] = "Moderate"
            elif improvements_count > 0:
                comparison["overall_improvement"] = "Minor"
            else:
                comparison["overall_improvement"] = "None or Negative"
            
            return comparison
            
        except Exception as e:
            print(f"Error comparing fairness metrics: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.evaluation_results:
            return {"error": "No evaluation has been run yet"}
        
        return {
            "dataset": self.current_dataset,
            "stages_completed": len(self.evaluation_results.get("stages", {})),
            "has_recommendations": "recommendations" in self.evaluation_results.get("stages", {})
        }




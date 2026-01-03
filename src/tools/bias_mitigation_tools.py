import pandas as pd
import numpy as np
from tools.tool import Tool
from tools.tool_manager import ToolManager
import os
import warnings
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

warnings.simplefilter(action='ignore', category=Warning)

class BiasMitigationTools(ToolManager):
    def __init__(self):
        super().__init__()
        
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
        self.tool_apply_reweighting = Tool(
            name="apply_reweighting",
            function=self.apply_reweighting,
            description="Apply reweighting technique to balance dataset. Creates sample weights to give more importance to underrepresented groups.",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"},
                    "target_column": {"type": "string", "description": "Target column name"},
                    "sensitive_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of sensitive columns to consider"
                    },
                    "output_dir": {"type": "string", "description": "Directory to save the results"}
                },
                "required": ["dataset_name", "target_column", "sensitive_columns", "output_dir"]
            }
        )
        
        self.tool_apply_smote = Tool(
            name="apply_smote",
            function=self.apply_smote,
            description="Apply SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for minority classes.",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"},
                    "target_column": {"type": "string", "description": "Target column name"},
                    "k_neighbors": {
                        "type": "integer",
                        "description": "Number of nearest neighbors for SMOTE (default: 5)",
                        "default": 5
                    },
                    "sampling_strategy": {
                        "type": "string",
                        "description": "Sampling strategy: 'auto', 'minority', 'not majority', 'all'",
                        "default": "auto"
                    },
                    "output_dir": {"type": "string", "description": "Directory to save the results"}
                },
                "required": ["dataset_name", "target_column", "output_dir"]
            }
        )
        
        self.tool_apply_oversampling = Tool(
            name="apply_random_oversampling",
            function=self.apply_oversampling,
            description="Apply random oversampling to duplicate samples from minority classes.",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"},
                    "target_column": {"type": "string", "description": "Target column name"},
                    "sampling_strategy": {
                        "type": "string",
                        "description": "Sampling strategy: 'auto', 'minority', 'not majority', 'all'",
                        "default": "auto"
                    },
                    "output_dir": {"type": "string", "description": "Directory to save the results"}
                },
                "required": ["dataset_name", "target_column", "output_dir"]
            }
        )
        
        self.tool_apply_undersampling = Tool(
            name="apply_random_undersampling",
            function=self.apply_undersampling,
            description="Apply random undersampling to reduce samples from majority classes.",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"},
                    "target_column": {"type": "string", "description": "Target column name"},
                    "sampling_strategy": {
                        "type": "string",
                        "description": "Sampling strategy: 'auto', 'not minority', 'majority', 'all'",
                        "default": "auto"
                    },
                    "output_dir": {"type": "string", "description": "Directory to save the results"}
                },
                "required": ["dataset_name", "target_column", "output_dir"]
            }
        )
        
        self.tool_compare_datasets = Tool(
            name="compare_datasets",
            function=self.compare_datasets,
            description="Compare original and mitigated datasets to evaluate the effectiveness of bias mitigation.",
            parameters={
                "type": "object",
                "properties": {
                    "original_dataset": {"type": "string", "description": "Name of the original dataset"},
                    "mitigated_dataset": {"type": "string", "description": "Name of the mitigated dataset"},
                    "target_column": {"type": "string", "description": "Target column name"},
                    "sensitive_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of sensitive columns to compare"
                    }
                },
                "required": ["original_dataset", "mitigated_dataset", "target_column", "sensitive_columns"]
            }
        )
        
        self.list_of_tools = [
            self.tool_apply_reweighting,
            self.tool_apply_smote,
            self.tool_apply_oversampling,
            self.tool_apply_undersampling,
            self.tool_compare_datasets
        ]
        self._build_tool_mappings()
    
    def _resolve_path(self, dataset_name: str) -> str:
        if not dataset_name.endswith('.csv'):
            dataset_name += '.csv'
        return os.path.join(self.data_dir, dataset_name)
    
    def apply_reweighting(self, dataset_name: str, target_column: str, 
                         sensitive_columns: list, output_dir: str) -> dict:
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            
            if target_column not in df.columns:
                return {"status": "error", "message": f"Target column '{target_column}' not found"}
            
            missing_cols = [col for col in sensitive_columns if col not in df.columns]
            if missing_cols:
                return {"status": "error", "message": f"Sensitive columns not found: {missing_cols}"}
            
            if len(sensitive_columns) == 1:
                group_col = sensitive_columns[0]
            else:
                group_col = "combined_group"
                df[group_col] = df[sensitive_columns].astype(str).agg('_'.join, axis=1)
            
            g_counts = df[group_col].value_counts()
            gy_counts = df.groupby([group_col, target_column]).size()
            num_classes = df[target_column].nunique()
            
            weights_map = {}
            for group_val in g_counts.index:
                for target_val in df[target_column].unique():
                    try:
                        count_gy = gy_counts.loc[(group_val, target_val)]
                    except KeyError:
                        count_gy = 0
                    
                    if count_gy > 0:
                        count_g = g_counts[group_val]
                        weight = (count_g / num_classes) / count_gy
                    else:
                        weight = 1.0 
                    
                    weights_map[(group_val, target_val)] = weight
            
            df['sample_weight'] = [
                weights_map.get((g, t), 1.0) 
                for g, t in zip(df[group_col], df[target_column])
            ]
            
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"{dataset_name.replace('.csv', '')}_reweighted.csv"
            output_path = os.path.join(output_dir, output_filename)
            df.to_csv(output_path, index=False)
            
            weights = df['sample_weight']
            weight_stats = {
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
                "mean": float(np.mean(weights)),
                "median": float(np.median(weights)),
                "std": float(np.std(weights))
            }
            
            distribution_before = df[target_column].value_counts().to_dict()
            
            weighted_dist = {}
            for target_val in df[target_column].unique():
                mask = df[target_column] == target_val
                weighted_count = df.loc[mask, 'sample_weight'].sum()
                weighted_dist[target_val] = weighted_count
            
            weighted_counts = list(weighted_dist.values())
            if len(weighted_counts) > 0 and min(weighted_counts) > 0:
                weighted_ratio = max(weighted_counts) / min(weighted_counts)
            else:
                weighted_ratio = 0
            
            return {
                "status": "success",
                "method": "Reweighting (Balanced + Fair)",
                "output_file": output_path,
                "original_rows": len(df),
                "new_rows": len(df), 
                "weight_statistics": weight_stats,
                "distribution_before": distribution_before,
                "distribution_after": weighted_dist, 
                "weighted_imbalance_ratio": round(weighted_ratio, 2),
                "sensitive_columns_used": sensitive_columns,
                "note": "Sample weights added as 'sample_weight' column. Weights calculated to balance target classes while preserving sensitive group distribution."
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Error in reweighting: {str(e)}"}
    
    def apply_smote(self, dataset_name: str, target_column: str, output_dir: str,
                   k_neighbors: int = 5, sampling_strategy: str = "auto") -> dict:
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            
            if target_column not in df.columns:
                return {"status": "error", "message": f"Target column '{target_column}' not found"}
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            X_encoded = X.copy()
            
            encodings = {}
            for col in categorical_cols:
                X_encoded[col] = X[col].astype('category').cat.codes
                encodings[col] = dict(enumerate(X[col].astype('category').cat.categories))
            
            if y.dtype == 'object':
                y_encoded = y.astype('category').cat.codes
                target_encoding = dict(enumerate(y.astype('category').cat.categories))
            else:
                y_encoded = y
                target_encoding = None
            
            distribution_before = y.value_counts().to_dict()
            
            smote = SMOTE(k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_encoded, y_encoded)
            
            if target_encoding:
                y_resampled = pd.Series(y_resampled).map(target_encoding)
            
            df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            for col in categorical_cols:
                df_resampled[col] = df_resampled[col].round().astype(int).map(encodings[col])
            
            df_resampled[target_column] = y_resampled
            
            distribution_after = pd.Series(y_resampled).value_counts().to_dict()
            
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"{dataset_name.replace('.csv', '')}_smote.csv"
            output_path = os.path.join(output_dir, output_filename)
            df_resampled.to_csv(output_path, index=False)
            
            return {
                "status": "success",
                "method": "SMOTE",
                "output_file": output_path,
                "original_rows": len(df),
                "new_rows": len(df_resampled),
                "rows_added": len(df_resampled) - len(df),
                "distribution_before": distribution_before,
                "distribution_after": distribution_after,
                "k_neighbors": k_neighbors,
                "sampling_strategy": sampling_strategy
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def apply_oversampling(self, dataset_name: str, target_column: str, output_dir: str,
                          sampling_strategy: str = "auto") -> dict:
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            
            if target_column not in df.columns:
                return {"status": "error", "message": f"Target column '{target_column}' not found"}
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            distribution_before = y.value_counts().to_dict()
            
            ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            
            df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            df_resampled[target_column] = y_resampled
            
            distribution_after = pd.Series(y_resampled).value_counts().to_dict()
            
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"{dataset_name.replace('.csv', '')}_oversampled.csv"
            output_path = os.path.join(output_dir, output_filename)
            df_resampled.to_csv(output_path, index=False)
            
            return {
                "status": "success",
                "method": "Random Oversampling",
                "output_file": output_path,
                "original_rows": len(df),
                "new_rows": len(df_resampled),
                "rows_added": len(df_resampled) - len(df),
                "distribution_before": distribution_before,
                "distribution_after": distribution_after,
                "sampling_strategy": sampling_strategy
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def apply_undersampling(self, dataset_name: str, target_column: str, output_dir: str,
                           sampling_strategy: str = "auto") -> dict:
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            
            if target_column not in df.columns:
                return {"status": "error", "message": f"Target column '{target_column}' not found"}
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            distribution_before = y.value_counts().to_dict()
            
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            
            df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            df_resampled[target_column] = y_resampled
            
            distribution_after = pd.Series(y_resampled).value_counts().to_dict()
            
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"{dataset_name.replace('.csv', '')}_undersampled.csv"
            output_path = os.path.join(output_dir, output_filename)
            df_resampled.to_csv(output_path, index=False)
            
            return {
                "status": "success",
                "method": "Random Undersampling",
                "output_file": output_path,
                "original_rows": len(df),
                "new_rows": len(df_resampled),
                "rows_removed": len(df) - len(df_resampled),
                "distribution_before": distribution_before,
                "distribution_after": distribution_after,
                "sampling_strategy": sampling_strategy
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def compare_datasets(self, original_dataset: str, mitigated_dataset: str,
                        target_column: str, sensitive_columns: list) -> dict:
        try:
            orig_path = self._resolve_path(original_dataset)
            mit_path = mitigated_dataset if os.path.exists(mitigated_dataset) else self._resolve_path(mitigated_dataset)
            
            df_orig = pd.read_csv(orig_path)
            df_mit = pd.read_csv(mit_path)
            
            has_weights = 'sample_weight' in df_mit.columns
            
            comparison = {
                "status": "success",
                "dataset_size": {
                    "original": len(df_orig),
                    "mitigated": len(df_mit),
                    "difference": len(df_mit) - len(df_orig),
                    "percentage_change": ((len(df_mit) - len(df_orig)) / len(df_orig) * 100)
                },
                "target_distribution": {},
                "uses_weights": has_weights
            }
            
            orig_dist = df_orig[target_column].value_counts()
            
            if has_weights:
                mit_dist_counts = {}
                for target_val in df_mit[target_column].unique():
                    mask = df_mit[target_column] == target_val
                    weighted_count = df_mit.loc[mask, 'sample_weight'].sum()
                    mit_dist_counts[target_val] = weighted_count
                
                total_weighted = sum(mit_dist_counts.values())
                
                for value in orig_dist.index:
                    orig_count = orig_dist.get(value, 0)
                    mit_weighted_count = mit_dist_counts.get(value, 0)
                    orig_pct = (orig_count / len(df_orig) * 100)
                    mit_weighted_pct = (mit_weighted_count / total_weighted * 100) if total_weighted > 0 else 0
                    
                    comparison["target_distribution"][str(value)] = {
                        "original_count": int(orig_count),
                        "original_percentage": round(orig_pct, 2),
                        "mitigated_weighted_count": round(mit_weighted_count, 2),
                        "mitigated_weighted_percentage": round(mit_weighted_pct, 2),
                        "weighted_change": round(mit_weighted_count - orig_count, 2),
                        "percentage_point_change": round(mit_weighted_pct - orig_pct, 2)
                    }
                
                weighted_values = list(mit_dist_counts.values())
                if len(weighted_values) > 0 and min(weighted_values) > 0:
                    mit_ratio = max(weighted_values) / min(weighted_values)
                else:
                    mit_ratio = 0
                    
            else:
                mit_dist = df_mit[target_column].value_counts()
                
                for value in orig_dist.index:
                    orig_count = orig_dist.get(value, 0)
                    mit_count = mit_dist.get(value, 0)
                    orig_pct = (orig_count / len(df_orig) * 100)
                    mit_pct = (mit_count / len(df_mit) * 100)
                    
                    comparison["target_distribution"][str(value)] = {
                        "original_count": int(orig_count),
                        "original_percentage": round(orig_pct, 2),
                        "mitigated_count": int(mit_count),
                        "mitigated_percentage": round(mit_pct, 2),
                        "change": int(mit_count - orig_count),
                        "percentage_point_change": round(mit_pct - orig_pct, 2)
                    }
                
                mit_values = df_mit[target_column].value_counts().values
                if len(mit_values) > 0 and min(mit_values) > 0:
                    mit_ratio = max(mit_values) / min(mit_values)
                else:
                    mit_ratio = 0
            
            comparison["sensitive_attributes"] = {}
            for col in sensitive_columns:
                if col in df_orig.columns and col in df_mit.columns:
                    orig_dist = df_orig[col].value_counts()
                    mit_dist = df_mit[col].value_counts()
                    
                    col_comparison = {}
                    all_values = set(orig_dist.index) | set(mit_dist.index)
                    
                    for value in all_values:
                        orig_count = orig_dist.get(value, 0)
                        mit_count = mit_dist.get(value, 0)
                        orig_pct = (orig_count / len(df_orig) * 100)
                        mit_pct = (mit_count / len(df_mit) * 100)
                        
                        col_comparison[str(value)] = {
                            "original_count": int(orig_count),
                            "original_percentage": round(orig_pct, 2),
                            "mitigated_count": int(mit_count),
                            "mitigated_percentage": round(mit_pct, 2),
                            "change": int(mit_count - orig_count)
                        }
                    
                    comparison["sensitive_attributes"][col] = col_comparison
            
            orig_values = df_orig[target_column].value_counts().values
            orig_ratio = max(orig_values) / min(orig_values) if len(orig_values) > 0 and min(orig_values) > 0 else 0
            
            comparison["imbalance_metrics"] = {
                "original_imbalance_ratio": round(orig_ratio, 2),
                "mitigated_imbalance_ratio": round(mit_ratio, 2),
                "improvement": "Yes" if mit_ratio < orig_ratio else "No",
                "uses_sample_weights": has_weights
            }
            
            if has_weights:
                comparison["imbalance_metrics"]["note"] = "Mitigated ratio calculated using sample weights. The actual improvement will be realized during model training when weights are applied."
            
            return comparison
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

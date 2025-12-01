import pandas as pd
from tools.tool import Tool
from tools.tool_manager import ToolManager
import os

class FairnessTools(ToolManager):    
    def __init__(self):
        super().__init__()
        
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
        self.tool_analyze_fairness = Tool(
            name="analyze_fairness",
            function=self.analyze_fairness,
            description="Comprehensive analysis of dataset for fairness issues, imbalanced data, sensitive attributes, and data quality",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset (e.g., 'adult-all' or 'adult-all.csv')"}
                },
                "required": ["dataset_name"]
            }
        )
        
        self.tool_check_missing = Tool(
            name="check_missing_data",
            function=self.check_missing_data,
            description="Analyze missing data in the dataset",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"}
                },
                "required": ["dataset_name"]
            }
        )
        
        self.tool_detect_sensitive = Tool(
            name="detect_sensitive_attributes",
            function=self.detect_sensitive_attributes,
            description="Detect sensitive/protected attributes in the dataset",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"}
                },
                "required": ["dataset_name"]
            }
        )
        
        self.tool_check_imbalance = Tool(
            name="check_class_imbalance",
            function=self.check_class_imbalance,
            description="Check for class imbalance in categorical features",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"}
                },
                "required": ["dataset_name"]
            }
        )
        
        self.tool_load_dataset = Tool(
            name="load_dataset",
            function=self.load_dataset,
            description="Load a CSV dataset from the data directory. Returns dataset info and preview.",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset to load (with or without .csv extension)"}
                },
                "required": ["dataset_name"]
            }
        )
        
        self.tool_get_dataset_preview = Tool(
            name="get_dataset_preview",
            function=self.get_dataset_preview,
            description="Get detailed preview of dataset including all column names, types, sample values, and statistics. Use this to understand the dataset structure before analysis.",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"}
                },
                "required": ["dataset_name"]
            }
        )
        
        self.tool_analyze_sensitive = Tool(
            name="analyze_sensitive_column",
            function=self.analyze_sensitive_column,
            description="Analyze a specific column for sensitive attributes and fairness concerns. Provides distribution and statistical analysis.",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"},
                    "column_name": {"type": "string", "description": "Name of the column to analyze"}
                },
                "required": ["dataset_name", "column_name"]
            }
        )
        
        self.list_of_tools = [
            self.tool_load_dataset,
            self.tool_get_dataset_preview,
            self.tool_analyze_fairness,
            self.tool_check_missing,
            self.tool_detect_sensitive,
            self.tool_analyze_sensitive,
            self.tool_check_imbalance
        ]
        self._build_tool_mappings()
    
    def load_dataset(self, dataset_name: str) -> dict:
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            return {
                "status": "success",
                "dataset_name": dataset_name,
                "path": path,
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head(3).to_dict(orient="records")
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_dataset_preview(self, dataset_name: str) -> dict:
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            
            column_info = []
            for col in df.columns:
                col_data = {
                    "name": col,
                    "type": str(df[col].dtype),
                    "unique_values": int(df[col].nunique()),
                    "null_count": int(df[col].isnull().sum()),
                    "sample_values": df[col].dropna().head(5).tolist()
                }
                
                if df[col].dtype in ['int64', 'float64']:
                    col_data["stats"] = {
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean())
                    }
                elif df[col].dtype == 'object':
                    top_values = df[col].value_counts().head(3)
                    col_data["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                
                column_info.append(col_data)
            
            return {
                "status": "success",
                "dataset_name": dataset_name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_details": column_info
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def analyze_sensitive_column(self, dataset_name: str, column_name: str) -> dict:
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            
            if column_name not in df.columns:
                return {"status": "error", "message": f"Column '{column_name}' not found"}
            
            col_data = df[column_name]
            result = {
                "status": "success",
                "column": column_name,
                "type": str(col_data.dtype),
                "unique_values": int(col_data.nunique()),
                "null_count": int(col_data.isnull().sum())
            }
            
            if col_data.dtype == 'object' or col_data.nunique() < 50:
                value_counts = col_data.value_counts()
                proportions = (value_counts / len(df) * 100).round(2)
                result["distribution"] = {str(k): {"count": int(v), "percentage": float(proportions[k])} 
                                         for k, v in value_counts.head(20).items()}
                result["imbalance_ratio"] = float(proportions.iloc[0] / proportions.iloc[-1]) if len(proportions) > 1 else 1.0
            else:
                result["stats"] = {
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std())
                }
            
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _resolve_path(self, dataset_name: str) -> str:
        if dataset_name.endswith('.csv'):
            dataset_name = dataset_name[:-4]
        
        possible_paths = [
            os.path.join(self.data_dir, f"{dataset_name}.csv"),
            os.path.join(self.data_dir, dataset_name), 
            dataset_name,  
            f"{dataset_name}.csv"  
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found dataset at: {path}")
                return path
        
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found. Tried:\n" + 
            "\n".join(f"  - {p}" for p in possible_paths) +
            f"\n\nData directory: {self.data_dir}" +
            f"\n\nAvailable datasets: {self._list_available_datasets()}"
        )
    
    def _list_available_datasets(self) -> str:
        try:
            if os.path.exists(self.data_dir):
                files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
                return ", ".join(files) if files else "No CSV files found"
            return f"Data directory not found: {self.data_dir}"
        except Exception as e:
            return f"Unable to list datasets: {str(e)}"
    
    
    def analyze_fairness(self, dataset_name: str):
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            
            report = {
                "status": "success",
                "dataset": dataset_name,
                "path": path,
                "dataset_info": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "columns": list(df.columns)
                },
                "issues": []
            }
            
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df) * 100).round(2)
            
            if missing_data.sum() > 0:
                missing_info = []
                for col in missing_data[missing_data > 0].index:
                    severity = "critical" if missing_pct[col] > 50 else "high" if missing_pct[col] > 20 else "medium"
                    missing_info.append({
                        "column": col,
                        "missing_count": int(missing_data[col]),
                        "missing_percentage": float(missing_pct[col]),
                        "severity": severity
                    })
                
                report["issues"].append({
                    "type": "missing_data",
                    "severity": "critical" if missing_pct.max() > 50 else "high" if missing_pct.max() > 20 else "medium",
                    "total_missing": int(missing_data.sum()),
                    "details": missing_info
                })
            
            # Basic categorical column analysis (no keyword matching)
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            imbalance_issues = []
            
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                proportions = (value_counts / len(df) * 100).round(2)
                
                if proportions.iloc[0] > 85:
                    imbalance_issues.append({
                        "column": col,
                        "severity": "critical",
                        "dominant_value": str(proportions.index[0]),
                        "dominant_pct": float(proportions.iloc[0])
                    })
            
            if imbalance_issues:
                report["issues"].append({
                    "type": "class_imbalance",
                    "severity": "critical",
                    "details": imbalance_issues
                })
            
            report["summary"] = {
                "total_issues": len(report["issues"]),
                "critical_issues": sum(1 for i in report["issues"] if i["severity"] == "critical"),
                "categorical_columns": len(categorical_cols),
                "note": "Sensitive attribute detection is agent-driven, not keyword-based"
            }
            
            return report
            
        except FileNotFoundError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": f"Error analyzing dataset: {str(e)}"}
    
    def check_missing_data(self, dataset_name: str):
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df) * 100).round(2)
            
            result = {
                "status": "success",
                "dataset": dataset_name,
                "total_rows": len(df),
                "columns_with_missing": int((missing_data > 0).sum()),
                "details": []
            }
            
            for col in missing_data[missing_data > 0].index:
                result["details"].append({
                    "column": col,
                    "missing_count": int(missing_data[col]),
                    "missing_percentage": float(missing_pct[col])
                })
            
            return result
            
        except FileNotFoundError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def detect_sensitive_attributes(self, dataset_name: str):
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            
            # Return all columns with their characteristics for agent analysis
            # No keyword matching - let the agent decide what's sensitive
            column_analysis = []
            
            for col in df.columns:
                col_info = {
                    "column": col,
                    "type": str(df[col].dtype),
                    "unique_values": int(df[col].nunique()),
                    "null_count": int(df[col].isnull().sum()),
                    "sample_values": df[col].dropna().head(5).tolist()
                }
                
                # Add distribution for categorical/low cardinality columns
                if df[col].dtype == 'object' or df[col].nunique() < 50:
                    value_counts = df[col].value_counts()
                    proportions = (value_counts / len(df) * 100).round(2)
                    col_info["top_values"] = {str(k): float(v) for k, v in proportions.head(5).items()}
                
                column_analysis.append(col_info)
            
            return {
                "status": "success",
                "dataset": dataset_name,
                "total_columns": len(column_analysis),
                "columns": column_analysis,
                "note": "Agent should analyze these columns to identify sensitive attributes based on names, values, and distributions"
            }
            
        except FileNotFoundError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def check_class_imbalance(self, dataset_name: str):
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            imbalances = []
            
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                proportions = (value_counts / len(df) * 100).round(2)
                
                # Use 65% threshold to catch imbalances like Sex (66.85%)
                if proportions.iloc[0] > 65:
                    imbalances.append({
                        "column": col,
                        "dominant_value": str(proportions.index[0]),
                        "dominant_percentage": float(proportions.iloc[0]),
                        "distribution": proportions.head(5).to_dict()
                    })
            
            return {
                "status": "success",
                "dataset": dataset_name,
                "imbalanced_columns": len(imbalances),
                "details": imbalances
            }
            
        except FileNotFoundError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}
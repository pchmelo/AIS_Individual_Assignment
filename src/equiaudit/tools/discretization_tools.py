import pandas as pd
import numpy as np
import os


class DiscretizationTools:
    """Utility class for discretizing continuous sensitive attributes into categorical bins.

    This is NOT a ToolManager — it is a plain helper used directly by the
    DiscretizationStage. It does not need to be registered in the tools config.
    """

    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    # ------------------------------------------------------------------
    # Path resolution (mirrors FairnessTools)
    # ------------------------------------------------------------------

    def _resolve_path(self, dataset_name: str) -> str:
        if dataset_name.endswith(".csv"):
            dataset_name = dataset_name[:-4]

        env_dataset = os.environ.get("FAIRNESS_DATASET_PATH", "")
        if env_dataset and os.path.exists(env_dataset):
            env_stem = os.path.splitext(os.path.basename(env_dataset))[0]
            if dataset_name == env_stem or dataset_name == os.path.basename(env_dataset):
                return env_dataset

        possible_paths = [
            os.path.join(self.data_dir, f"{dataset_name}.csv"),
            os.path.join(self.data_dir, dataset_name),
            dataset_name,
            f"{dataset_name}.csv",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(f"Dataset '{dataset_name}' not found.")

    # ------------------------------------------------------------------
    # 1. Identify continuous sensitive columns
    # ------------------------------------------------------------------

    def identify_continuous_sensitive(
        self,
        dataset_name: str,
        sensitive_columns: list,
        unique_threshold: int = 10,
    ) -> dict:
        """Return statistics for every sensitive column classified as continuous."""
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)

            continuous_columns = []
            discrete_columns = []

            for col in sensitive_columns:
                if col not in df.columns:
                    continue

                series = df[col]
                is_numeric = pd.api.types.is_numeric_dtype(series)
                n_unique = int(series.nunique())

                if is_numeric and n_unique > unique_threshold:
                    stats = {
                        "column_name": col,
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "mean": round(float(series.mean()), 4),
                        "median": round(float(series.median()), 4),
                        "std": round(float(series.std()), 4),
                        "unique_count": n_unique,
                        "total_count": int(len(series.dropna())),
                        "null_count": int(series.isnull().sum()),
                        "sample_values": sorted(series.dropna().unique()[:20].tolist()),
                    }
                    continuous_columns.append(stats)
                else:
                    discrete_columns.append(col)

            return {
                "status": "success",
                "dataset_name": dataset_name,
                "unique_threshold": unique_threshold,
                "continuous_columns": continuous_columns,
                "discrete_columns": discrete_columns,
                "total_sensitive": len(sensitive_columns),
                "num_continuous": len(continuous_columns),
                "num_discrete": len(discrete_columns),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # 2. Discretize with agent-decided bin edges
    # ------------------------------------------------------------------

    def discretize_column_auto(
        self,
        dataset_name: str,
        column_name: str,
        bin_edges: list,
        labels: list = None,
    ) -> dict:
        """Discretize *column_name* using explicit *bin_edges*."""
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)

            if column_name not in df.columns:
                return {"status": "error", "message": f"Column '{column_name}' not found"}

            bin_edges = sorted(set(bin_edges))
            if len(bin_edges) < 2:
                return {"status": "error", "message": "Need at least 2 bin edges"}

            n_bins = len(bin_edges) - 1
            if labels and len(labels) != n_bins:
                return {
                    "status": "error",
                    "message": f"Expected {n_bins} labels but got {len(labels)}",
                }

            if not labels:
                labels = [
                    f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(n_bins)
                ]

            # Extend edges slightly to include min/max values on borders
            edges = list(bin_edges)
            col_min = float(df[column_name].min())
            col_max = float(df[column_name].max())
            if edges[0] > col_min:
                edges[0] = col_min
            if edges[-1] < col_max:
                edges[-1] = col_max + 0.001

            # Preserve original values in a new column
            original_col = f"{column_name}_original"
            df[original_col] = df[column_name]

            df[column_name] = pd.cut(
                df[column_name],
                bins=edges,
                labels=labels,
                include_lowest=True,
            )
            # Convert to string for downstream compatibility
            df[column_name] = df[column_name].astype(str)

            # Save modified dataset
            output_path = self._save_discretized(df, dataset_name)

            # Build distribution
            dist = df[column_name].value_counts().to_dict()
            dist = {str(k): int(v) for k, v in dist.items()}

            return {
                "status": "success",
                "column": column_name,
                "method": "auto",
                "bin_edges": bin_edges,
                "labels": labels,
                "distribution": dist,
                "output_dataset": os.path.basename(output_path),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # 3. Discretize with equal-width or equal-frequency
    # ------------------------------------------------------------------

    def discretize_column_manual(
        self,
        dataset_name: str,
        column_name: str,
        method: str,
        number_of_bins: int,
    ) -> dict:
        """Discretize *column_name* using equal-width or equal-frequency bins."""
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)

            if column_name not in df.columns:
                return {"status": "error", "message": f"Column '{column_name}' not found"}

            # Preserve original values in a new column
            original_col = f"{column_name}_original"
            df[original_col] = df[column_name]

            if method == "equal_width":
                df[column_name], bin_edges = pd.cut(
                    df[column_name],
                    bins=number_of_bins,
                    retbins=True,
                    include_lowest=True,
                )
                intervals = df[column_name].cat.categories
                labels = [str(iv) for iv in intervals]
                bin_edges = bin_edges.tolist()

            elif method == "equal_frequency":
                df[column_name], bin_edges = pd.qcut(
                    df[column_name],
                    q=number_of_bins,
                    retbins=True,
                    duplicates="drop",
                )
                intervals = df[column_name].cat.categories
                labels = [str(iv) for iv in intervals]
                bin_edges = bin_edges.tolist()
            else:
                return {
                    "status": "error",
                    "message": f"Unknown method '{method}'. Use 'equal_width' or 'equal_frequency'.",
                }

            # Convert to string for downstream compatibility
            df[column_name] = df[column_name].astype(str)

            # Save modified dataset
            output_path = self._save_discretized(df, dataset_name)

            # Build distribution
            dist = df[column_name].value_counts().to_dict()
            dist = {str(k): int(v) for k, v in dist.items()}

            return {
                "status": "success",
                "column": column_name,
                "method": method,
                "number_of_bins": number_of_bins,
                "bin_edges": bin_edges,
                "labels": labels,
                "distribution": dist,
                "output_dataset": os.path.basename(output_path),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_discretized(self, df: pd.DataFrame, dataset_name: str) -> str:
        """Save the discretized DataFrame, returning the output path."""
        stem = dataset_name.replace(".csv", "")
        # Always overwrite the same discretized file (idempotent across columns)
        out_name = f"{stem}_discretized.csv"
        out_path = os.path.join(self.data_dir, out_name)
        df.to_csv(out_path, index=False)
        return out_path

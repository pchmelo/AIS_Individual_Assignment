import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from scipy import stats

try:
    from ydata_profiling import ProfileReport 
except ImportError:
    ProfileReport = None

try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False


class DatasetEvaluator:
    def __init__(self, csv_path: str, target_column: Optional[str] = None,
                 sensitive_attributes: Optional[List[str]] = None):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, na_values=['?', 'NA', 'N/A', '', 'null', 'NULL'])
        self.target_column = target_column or self._detect_target_column()
        self.sensitive_attributes = sensitive_attributes or self._detect_sensitive_attributes()
        self.report = {}
        
    def _detect_target_column(self) -> Optional[str]:
        common_targets = ['target', 'label', 'class', 'outcome', 'y', 'income', 
                         'churn', 'default', 'fraud', 'diagnosis']
        
        for col in self.df.columns:
            if col.lower() in common_targets:
                return col
        
        return self.df.columns[-1] if len(self.df.columns) > 0 else None
    
    def _detect_sensitive_attributes(self) -> List[str]:
        sensitive_keywords = ['sex', 'gender', 'race', 'ethnicity', 'age', 'religion',
                             'nationality', 'disability', 'sexual_orientation', 'marital']
        
        detected = []
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in sensitive_keywords):
                detected.append(col)
        
        return detected
    
    def evaluate_all(self) -> Dict[str, Any]:
        print("Starting comprehensive dataset evaluation...")
        
        self.report['basic_info'] = self.analyze_basic_info()
        self.report['missing_data'] = self.analyze_missing_data()
        self.report['data_quality'] = self.analyze_data_quality()
        self.report['class_balance'] = self.analyze_class_balance()
        self.report['feature_analysis'] = self.analyze_features()
        self.report['sensitive_labels'] = self.detect_sensitive_labels()
        self.report['fairness_analysis'] = self.analyze_fairness()
        self.report['statistical_tests'] = self.run_statistical_tests()
        self.report['data_drift_risk'] = self.assess_data_drift_risk()
        self.report['automated_profiling'] = self.generate_automated_profile()
        self.report['overall_quality_score'] = self.calculate_quality_score()
        
        print("Evaluation complete!")
        return self.report
    
    def analyze_basic_info(self) -> Dict[str, Any]:
        return {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'n_features': len(self.df.columns) - (1 if self.target_column else 0),
            'target_column': self.target_column,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object', 'category']).columns)
        }
    
    def analyze_missing_data(self) -> Dict[str, Any]:
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df)) * 100
        
        missing_info = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_pct
        }).sort_values('missing_percentage', ascending=False)
        
        missing_info = missing_info[missing_info['missing_count'] > 0]
        
        missing_pattern = self.df.isnull().astype(int)
        pattern_counts = missing_pattern.value_counts().head(10)
        
        return {
            'total_missing_cells': int(self.df.isnull().sum().sum()),
            'missing_percentage': float((self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100),
            'columns_with_missing': missing_info.to_dict('index'),
            'complete_rows': int((~self.df.isnull().any(axis=1)).sum()),
            'rows_with_missing': int(self.df.isnull().any(axis=1).sum()),
            'recommendation': self._missing_data_recommendation(missing_info)
        }
    
    def _missing_data_recommendation(self, missing_info: pd.DataFrame) -> str:
        if len(missing_info) == 0:
            return "No missing data detected. Dataset is complete."
        
        max_missing = missing_info['missing_percentage'].max()
        
        if max_missing > 50:
            return "CRITICAL: Some columns have >50% missing data. Consider removing these columns."
        elif max_missing > 20:
            return "WARNING: Significant missing data detected. Use imputation or consider impact on model."
        else:
            return "Minor missing data detected. Simple imputation should suffice."
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        issues = []
        
        n_duplicates = self.df.duplicated().sum()
        if n_duplicates > 0:
            issues.append(f"Found {n_duplicates} duplicate rows ({n_duplicates/len(self.df)*100:.2f}%)")
        
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() == 1]
        if constant_cols:
            issues.append(f"Found {len(constant_cols)} constant columns: {constant_cols}")
        
        high_card_cols = []
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            if self.df[col].nunique() > len(self.df) * 0.5:
                high_card_cols.append(col)
        
        if high_card_cols:
            issues.append(f"High cardinality categorical columns: {high_card_cols}")
        
        outlier_info = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                outlier_info[col] = {
                    'count': int(outliers),
                    'percentage': float(outliers / len(self.df) * 100)
                }
        
        return {
            'n_duplicates': int(n_duplicates),
            'duplicate_percentage': float(n_duplicates / len(self.df) * 100),
            'constant_columns': constant_cols,
            'high_cardinality_columns': high_card_cols,
            'outliers': outlier_info,
            'quality_issues': issues,
            'quality_flag': 'GOOD' if len(issues) == 0 else 'NEEDS_ATTENTION'
        }
    
    def analyze_class_balance(self) -> Dict[str, Any]:
        if not self.target_column or self.target_column not in self.df.columns:
            return {'status': 'No target column identified'}
        
        value_counts = self.df[self.target_column].value_counts()
        value_props = self.df[self.target_column].value_counts(normalize=True)
        
        if len(value_counts) >= 2:
            imbalance_ratio = value_counts.max() / value_counts.min()
        else:
            imbalance_ratio = 1.0
        
        if imbalance_ratio < 1.5:
            balance_status = "BALANCED"
            recommendation = "Classes are well balanced."
        elif imbalance_ratio < 3:
            balance_status = "SLIGHTLY_IMBALANCED"
            recommendation = "ℹMinor imbalance. Consider stratified sampling."
        elif imbalance_ratio < 10:
            balance_status = "IMBALANCED"
            recommendation = "Significant imbalance. Use SMOTE, class weights, or oversampling."
        else:
            balance_status = "HIGHLY_IMBALANCED"
            recommendation = "CRITICAL: Severe class imbalance. Use advanced techniques (ADASYN, focal loss)."
        
        return {
            'target_column': self.target_column,
            'n_classes': int(len(value_counts)),
            'class_distribution': value_counts.to_dict(),
            'class_proportions': value_props.to_dict(),
            'imbalance_ratio': float(imbalance_ratio),
            'balance_status': balance_status,
            'recommendation': recommendation
        }
    
    def analyze_features(self) -> Dict[str, Any]:
        feature_stats = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == self.target_column:
                continue
            
            feature_stats[col] = {
                'type': 'numeric',
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'median': float(self.df[col].median()),
                'skewness': float(self.df[col].skew()),
                'kurtosis': float(self.df[col].kurtosis()),
                'n_unique': int(self.df[col].nunique())
            }
        
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col == self.target_column:
                continue
            
            feature_stats[col] = {
                'type': 'categorical',
                'n_unique': int(self.df[col].nunique()),
                'most_common': self.df[col].value_counts().head(3).to_dict(),
                'entropy': float(stats.entropy(self.df[col].value_counts()))
            }
        
        return feature_stats
    
    def detect_sensitive_labels(self) -> Dict[str, Any]:
        sensitive_patterns = {
            'personal_identifiers': ['id', 'name', 'email', 'phone', 'ssn', 'address', 'zip'],
            'protected_attributes': ['race', 'gender', 'sex', 'age', 'religion', 'disability', 
                                    'nationality', 'ethnicity', 'sexual_orientation'],
            'financial': ['salary', 'income', 'credit', 'loan', 'debt'],
            'health': ['disease', 'diagnosis', 'medical', 'health', 'symptom', 'medication']
        }
        
        detected = {category: [] for category in sensitive_patterns}
        
        for col in self.df.columns:
            col_lower = col.lower()
            for category, patterns in sensitive_patterns.items():
                if any(pattern in col_lower for pattern in patterns):
                    detected[category].append(col)
        
        detected = {k: v for k, v in detected.items() if v}
        
        warning = ""
        if detected:
            warning = "PRIVACY WARNING: Sensitive attributes detected. Ensure proper anonymization and compliance with regulations (GDPR, HIPAA, etc.)"
        
        return {
            'sensitive_columns_detected': detected,
            'warning': warning,
            'recommendation': "Review data handling policies and consider pseudonymization or removal of identifiable information."
        }
    
    def analyze_fairness(self) -> Dict[str, Any]:
        if not self.sensitive_attributes or not self.target_column:
            return {'status': 'No sensitive attributes or target column identified'}
        
        fairness_metrics = {}
        
        for attr in self.sensitive_attributes:
            if attr not in self.df.columns:
                continue
            
            grouped = self.df.groupby(attr)[self.target_column].value_counts(normalize=True).unstack(fill_value=0)
            
            if len(grouped) >= 2:
                positive_rates = {}
                for group in grouped.index:
                    if isinstance(grouped.loc[group], pd.Series):
                        positive_rate = grouped.loc[group].max()
                    else:
                        positive_rate = float(grouped.loc[group])
                    positive_rates[str(group)] = float(positive_rate)
                
                max_rate = max(positive_rates.values())
                min_rate = min(positive_rates.values())
                disparity = max_rate - min_rate
                
                if disparity < 0.1:
                    fairness_level = "FAIR"
                elif disparity < 0.2:
                    fairness_level = "MINOR_DISPARITY"
                else:
                    fairness_level = "SIGNIFICANT_DISPARITY"
                
                fairness_metrics[attr] = {
                    'positive_rates_by_group': positive_rates,
                    'statistical_parity_difference': float(disparity),
                    'fairness_level': fairness_level,
                    'group_sizes': self.df[attr].value_counts().to_dict()
                }
        
        if AIF360_AVAILABLE:
            try:
                aif360_metrics = self._compute_aif360_metrics()
                fairness_metrics['aif360_advanced_metrics'] = aif360_metrics
            except Exception as e:
                fairness_metrics['aif360_error'] = f"Could not compute AIF360 metrics: {str(e)}"
        else:
            fairness_metrics['aif360_note'] = "Install aif360 for advanced fairness metrics"
        
        return fairness_metrics
    
    def _compute_aif360_metrics(self) -> Dict[str, Any]:
        if not AIF360_AVAILABLE:
            return {}
        
        metrics_results = {}
        
        df_copy = self.df.copy()
        
        target_values = df_copy[self.target_column].unique()
        if len(target_values) == 2:
            le_target = LabelEncoder()
            df_copy[self.target_column + '_encoded'] = le_target.fit_transform(df_copy[self.target_column])
            
            for attr in self.sensitive_attributes:
                if attr not in df_copy.columns:
                    continue
                
                try:
                    le_attr = LabelEncoder()
                    df_copy[attr + '_encoded'] = le_attr.fit_transform(df_copy[attr].fillna('Unknown'))
                    
                    aif_dataset = BinaryLabelDataset(
                        favorable_label=1,
                        unfavorable_label=0,
                        df=df_copy,
                        label_names=[self.target_column + '_encoded'],
                        protected_attribute_names=[attr + '_encoded']
                    )
                    
                    metric = BinaryLabelDatasetMetric(
                        aif_dataset,
                        unprivileged_groups=[{attr + '_encoded': 0}],
                        privileged_groups=[{attr + '_encoded': 1}]
                    )
                    
                    metrics_results[attr] = {
                        'disparate_impact': float(metric.disparate_impact()),
                        'statistical_parity_difference': float(metric.statistical_parity_difference()),
                        'consistency': float(metric.consistency()[0]) if hasattr(metric.consistency(), '__iter__') else float(metric.consistency()),
                        'base_rate_privileged': float(metric.base_rate(privileged=True)),
                        'base_rate_unprivileged': float(metric.base_rate(privileged=False))
                    }
                    
                    di = metrics_results[attr]['disparate_impact']
                    if 0.8 <= di <= 1.25:
                        metrics_results[attr]['disparate_impact_status'] = "ACCEPTABLE (80% rule satisfied)"
                    else:
                        metrics_results[attr]['disparate_impact_status'] = "CONCERNING (violates 80% rule)"
                    
                except Exception as e:
                    metrics_results[attr] = {'error': str(e)}
        
        return metrics_results
    
    def run_statistical_tests(self) -> Dict[str, Any]:
        tests = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        normality_tests = {}
        
        for col in numeric_cols[:10]:
            if len(self.df[col].dropna()) > 3:
                statistic, p_value = stats.shapiro(self.df[col].dropna().sample(min(5000, len(self.df))))
                normality_tests[col] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'is_normal': bool(p_value > 0.05)
                }
        
        tests['normality'] = normality_tests
        
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_matrix.iloc[i, j])
                        })
            
            tests['high_correlations'] = high_corr
        
        return tests
    
    def assess_data_drift_risk(self) -> Dict[str, Any]:
        risks = []
        
        date_cols = []
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower():
                date_cols.append(col)
        
        if not date_cols:
            risks.append("No temporal columns detected - cannot assess temporal drift risk")
        
        for col in self.df.select_dtypes(include=[np.number]).columns[:5]:
            cv = self.df[col].std() / (self.df[col].mean() + 1e-10)
            if cv > 2:
                risks.append(f"High coefficient of variation in {col} - may indicate unstable distribution")
        
        return {
            'temporal_columns': date_cols,
            'drift_risks': risks,
            'recommendation': "Implement monitoring for data drift in production" if not risks else "Address identified risks before deployment"
        }
    
    def generate_automated_profile(self) -> Dict[str, Any]:
        if ProfileReport is None:
            return {
                'status': 'ydata-profiling not installed',
                'note': 'Install ydata-profiling for comprehensive automated analysis: pip install ydata-profiling'
            }
        
        try:
            print("  Generating automated profile (this may take a moment)...")
            
            profile = ProfileReport(
                self.df,
                title="Dataset Profile Report",
                minimal=False,
                explorative=True,
                sensitive=len(self.sensitive_attributes) > 0,
                correlations={
                    "auto": {"calculate": True},
                    "pearson": {"calculate": True},
                    "spearman": {"calculate": True},
                    "kendall": {"calculate": False},
                    "phi_k": {"calculate": True},
                    "cramers": {"calculate": True},
                },
            )
            
            profile_dict = profile.get_description()
            
            result = {
                'status': 'success',
                'n_variables': profile_dict['table']['n_var'],
                'n_observations': profile_dict['table']['n'],
                'missing_cells': profile_dict['table']['n_cells_missing'],
                'missing_cells_pct': profile_dict['table']['p_cells_missing'],
                'duplicate_rows': profile_dict['table']['n_duplicates'],
                'duplicate_rows_pct': profile_dict['table']['p_duplicates'],
                'total_size_in_memory': profile_dict['table']['memory_size'],
                'variables_summary': {}
            }
            
            for var_name, var_data in profile_dict.get('variables', {}).items():
                if var_name in self.df.columns[:20]:
                    result['variables_summary'][var_name] = {
                        'type': var_data.get('type', 'unknown'),
                        'n_missing': var_data.get('n_missing', 0),
                        'p_missing': var_data.get('p_missing', 0),
                        'n_distinct': var_data.get('n_distinct', 0),
                        'is_unique': var_data.get('is_unique', False),
                    }
                    
                    if var_data.get('type') == 'Numeric':
                        result['variables_summary'][var_name].update({
                            'mean': var_data.get('mean'),
                            'std': var_data.get('std'),
                            'min': var_data.get('min'),
                            'max': var_data.get('max'),
                            'zeros_count': var_data.get('n_zeros', 0),
                        })
                    elif var_data.get('type') == 'Categorical':
                        result['variables_summary'][var_name].update({
                            'n_categories': var_data.get('n_unique', 0),
                        })
            
            if 'correlations' in profile_dict:
                high_correlations = []
                for corr_type, corr_data in profile_dict['correlations'].items():
                    if isinstance(corr_data, dict) and 'matrix' in corr_data:
                        pass
                result['correlation_analysis'] = 'Available in full report'
            
            if 'alerts' in profile_dict:
                result['alerts'] = profile_dict['alerts']
            
            result['note'] = 'Full HTML report available via profile.to_file()'
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'note': 'Automated profiling failed. Basic analysis still available.'
            }
    
    def calculate_quality_score(self) -> Dict[str, Any]:
        score = 100.0
        deductions = []
        
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        if missing_pct > 0:
            penalty = min(missing_pct * 0.5, 30)
            score -= penalty
            deductions.append(f"Missing data: -{penalty:.1f}")
        
        dup_pct = (self.df.duplicated().sum() / len(self.df)) * 100
        if dup_pct > 0:
            penalty = min(dup_pct * 0.3, 15)
            score -= penalty
            deductions.append(f"Duplicates: -{penalty:.1f}")
        
        if self.target_column and self.target_column in self.df.columns:
            value_counts = self.df[self.target_column].value_counts()
            if len(value_counts) >= 2:
                imbalance_ratio = value_counts.max() / value_counts.min()
                if imbalance_ratio > 3:
                    penalty = min((imbalance_ratio - 3) * 2, 20)
                    score -= penalty
                    deductions.append(f"Class imbalance: -{penalty:.1f}")
        
        if len(self.df) < 1000:
            penalty = 10
            score -= penalty
            deductions.append(f"Small sample size: -{penalty:.1f}")
        
        score = max(0, score)
        
        if score >= 90:
            grade = "A - EXCELLENT"
        elif score >= 80:
            grade = "B - GOOD"
        elif score >= 70:
            grade = "C - ACCEPTABLE"
        elif score >= 60:
            grade = "D - POOR"
        else:
            grade = "F - UNACCEPTABLE"
        
        return {
            'overall_score': float(score),
            'grade': grade,
            'deductions': deductions,
            'ready_for_ml': bool(score >= 70)
        }
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        if not self.report:
            self.evaluate_all()
        
        report_lines = [
            "=" * 80,
            "DATASET QUALITY EVALUATION REPORT",
            "=" * 80,
            f"\nDataset: {self.csv_path}",
            f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n" + "=" * 80,
            "\nBASIC INFORMATION",
            "-" * 80,
            f"Rows: {self.report['basic_info']['n_rows']:,}",
            f"Columns: {self.report['basic_info']['n_columns']}",
            f"Features: {self.report['basic_info']['n_features']}",
            f"Target Column: {self.report['basic_info']['target_column']}",
            f"Memory Usage: {self.report['basic_info']['memory_usage_mb']:.2f} MB",
            "\n" + "=" * 80,
            f"\nOVERALL QUALITY SCORE: {self.report['overall_quality_score']['overall_score']:.1f}/100",
            f"Grade: {self.report['overall_quality_score']['grade']}",
            f"Ready for ML: {'YES' if self.report['overall_quality_score']['ready_for_ml'] else 'NO'}",
        ]
        
        if self.report['overall_quality_score']['deductions']:
            report_lines.append("\nDeductions:")
            for deduction in self.report['overall_quality_score']['deductions']:
                report_lines.append(f"  • {deduction}")
        
        report_lines.extend([
            "\n" + "=" * 80,
            "\nKEY FINDINGS",
            "-" * 80,
        ])
        
        if 'target_column' in self.report['class_balance']:
            cb = self.report['class_balance']
            report_lines.extend([
                f"\nClass Balance: {cb['balance_status']}",
                f"  • Imbalance Ratio: {cb['imbalance_ratio']:.2f}",
                f"  • {cb['recommendation']}"
            ])
        
        md = self.report['missing_data']
        report_lines.extend([
            f"\nMissing Data: {md['missing_percentage']:.2f}%",
            f"  • {md['recommendation']}"
        ])
        
        if self.report['sensitive_labels']['warning']:
            report_lines.extend([
                f"\n{self.report['sensitive_labels']['warning']}",
                f"  • Detected: {list(self.report['sensitive_labels']['sensitive_columns_detected'].keys())}"
            ])
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def save_report_json(self, output_path: str):
        if not self.report:
            self.evaluate_all()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, default=str)


def evaluate_dataset(csv_path: str, 
                     target_column: Optional[str] = None,
                     sensitive_attributes: Optional[List[str]] = None,
                     output_json: Optional[str] = None,
                     output_txt: Optional[str] = None) -> Dict[str, Any]:

    evaluator = DatasetEvaluator(csv_path, target_column, sensitive_attributes)
    report = evaluator.evaluate_all()
    
    if output_json:
        evaluator.save_report_json(output_json)
    
    if output_txt:
        evaluator.generate_report(output_txt)
    else:
        print(evaluator.generate_report())
    
    return report


if __name__ == "__main__":
    report = evaluate_dataset(
        csv_path="adult.csv",
        target_column="income",
        sensitive_attributes=["sex", "race"],
        output_json="dataset_report.json",
        output_txt="dataset_report.txt"
    )
    
    print(f"\nEvaluation complete!")
    print(f"Overall Score: {report['overall_quality_score']['overall_score']:.1f}/100")
    print(f"Grade: {report['overall_quality_score']['grade']}")
"""
Data Validation and Quality Checks Module
Comprehensive data quality analysis and validation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataValidator:
    """
    Comprehensive data validation and quality checker
    """
    
    def __init__(self, df, target_column=None):
        """
        Args:
            df: pandas DataFrame to validate
            target_column: Name of target column (optional)
        """
        self.df = df.copy()
        self.target_column = target_column
        self.report = {}
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run all validation checks and return comprehensive report"""
        
        print("🔍 Running comprehensive data validation...")
        
        self.report = {
            "basic_info": self._check_basic_info(),
            "missing_values": self._check_missing_values(),
            "duplicates": self._check_duplicates(),
            "data_types": self._check_data_types(),
            "outliers": self._check_outliers(),
            "cardinality": self._check_cardinality(),
            "correlations": self._check_correlations(),
            "target_analysis": self._analyze_target() if self.target_column else None,
            "quality_score": 0,
            "recommendations": []
        }
        
        # Calculate quality score
        self.report["quality_score"] = self._calculate_quality_score()
        
        # Generate recommendations
        self.report["recommendations"] = self._generate_recommendations()
        
        print(f"✅ Validation complete! Quality Score: {self.report['quality_score']:.1f}/100")
        
        return self.report
    
    def _check_basic_info(self) -> Dict[str, Any]:
        """Basic dataset information"""
        return {
            "num_rows": len(self.df),
            "num_columns": len(self.df.columns),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
            "column_names": list(self.df.columns),
        }
    
    def _check_missing_values(self) -> Dict[str, Any]:
        """Check for missing values"""
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df)) * 100
        
        columns_with_missing = missing_pct[missing_pct > 0].to_dict()
        
        return {
            "total_missing_cells": int(self.df.isnull().sum().sum()),
            "pct_missing_cells": float((self.df.isnull().sum().sum() / self.df.size) * 100),
            "columns_with_missing": columns_with_missing,
            "severe_missing_cols": [col for col, pct in columns_with_missing.items() if pct > 50]
        }
    
    def _check_duplicates(self) -> Dict[str, Any]:
        """Check for duplicate rows"""
        num_duplicates = self.df.duplicated().sum()
        
        return {
            "num_duplicates": int(num_duplicates),
            "pct_duplicates": float((num_duplicates / len(self.df)) * 100),
            "has_duplicates": bool(num_duplicates > 0)
        }
    
    def _check_data_types(self) -> Dict[str, Any]:
        """Analyze data types"""
        type_counts = self.df.dtypes.value_counts().to_dict()
        
        # Convert dtype keys to strings
        type_counts = {str(k): int(v) for k, v in type_counts.items()}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            "type_distribution": type_counts,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "num_numeric": len(numeric_cols),
            "num_categorical": len(categorical_cols),
        }
    
    def _check_outliers(self) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        outlier_info = {}
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            
            if outliers > 0:
                outlier_info[col] = {
                    "count": int(outliers),
                    "pct": float((outliers / len(self.df)) * 100),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                }
        
        return {
            "columns_with_outliers": outlier_info,
            "total_outlier_columns": len(outlier_info),
        }
    
    def _check_cardinality(self) -> Dict[str, Any]:
        """Check cardinality of categorical columns"""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        cardinality_info = {}
        high_cardinality = []
        
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            unique_pct = (unique_count / len(self.df)) * 100
            
            cardinality_info[col] = {
                "unique_values": int(unique_count),
                "pct_unique": float(unique_pct),
            }
            
            # Flag high cardinality (>50% unique or >100 categories)
            if unique_pct > 50 or unique_count > 100:
                high_cardinality.append(col)
        
        return {
            "cardinality_per_column": cardinality_info,
            "high_cardinality_columns": high_cardinality,
        }
    
    def _check_correlations(self) -> Dict[str, Any]:
        """Check for highly correlated features"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {"high_correlations": []}
        
        corr_matrix = numeric_df.corr()
        
        # Find high correlations (>0.9)
        high_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.9:
                    high_corr.append({
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": float(corr_val),
                    })
        
        return {
            "high_correlations": high_corr,
            "num_high_correlations": len(high_corr),
        }
    
    def _analyze_target(self) -> Dict[str, Any]:
        """Analyze target variable"""
        if self.target_column not in self.df.columns:
            return {"error": "Target column not found"}
        
        target = self.df[self.target_column]
        
        analysis = {
            "column_name": self.target_column,
            "dtype": str(target.dtype),
            "missing_count": int(target.isnull().sum()),
            "missing_pct": float((target.isnull().sum() / len(target)) * 100),
        }
        
        # Numeric target
        if pd.api.types.is_numeric_dtype(target):
            analysis.update({
                "type": "numeric",
                "mean": float(target.mean()),
                "median": float(target.median()),
                "std": float(target.std()),
                "min": float(target.min()),
                "max": float(target.max()),
                "num_unique": int(target.nunique()),
            })
        
        # Categorical target
        else:
            value_counts = target.value_counts()
            analysis.update({
                "type": "categorical",
                "num_classes": int(target.nunique()),
                "class_distribution": value_counts.to_dict(),
                "is_balanced": bool(value_counts.max() / value_counts.min() < 3),
            })
        
        return analysis
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        
        # Deduct for missing values
        missing_pct = self.report["missing_values"]["pct_missing_cells"]
        score -= min(missing_pct, 30)  # Max 30 points deduction
        
        # Deduct for duplicates
        dup_pct = self.report["duplicates"]["pct_duplicates"]
        score -= min(dup_pct, 20)  # Max 20 points deduction
        
        # Deduct for outliers
        outlier_cols = self.report["outliers"]["total_outlier_columns"]
        total_numeric = self.report["data_types"]["num_numeric"]
        if total_numeric > 0:
            outlier_pct = (outlier_cols / total_numeric) * 100
            score -= min(outlier_pct / 2, 15)  # Max 15 points deduction
        
        # Deduct for high correlations
        high_corr = self.report["correlations"]["num_high_correlations"]
        score -= min(high_corr * 2, 10)  # Max 10 points deduction
        
        return max(score, 0)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Missing values
        if self.report["missing_values"]["pct_missing_cells"] > 5:
            recommendations.append(
                f"⚠️  {self.report['missing_values']['pct_missing_cells']:.1f}% of data is missing. "
                "Consider imputation or dropping columns with >50% missing."
            )
        
        # Duplicates
        if self.report["duplicates"]["has_duplicates"]:
            recommendations.append(
                f"🔄 {self.report['duplicates']['num_duplicates']} duplicate rows found. "
                "Consider removing duplicates."
            )
        
        # Outliers
        if self.report["outliers"]["total_outlier_columns"] > 0:
            recommendations.append(
                f"📊 {self.report['outliers']['total_outlier_columns']} columns have outliers. "
                "Consider capping, transformation, or removal."
            )
        
        # High cardinality
        if "high_cardinality_columns" in self.report["cardinality"]:
            high_card = self.report["cardinality"]["high_cardinality_columns"]
            if high_card:
                recommendations.append(
                    f"🗂️  {len(high_card)} high-cardinality columns detected. "
                    "Consider encoding strategies or grouping rare categories."
                )
        
        # High correlations
        if self.report["correlations"]["num_high_correlations"] > 0:
            recommendations.append(
                f"🔗 {self.report['correlations']['num_high_correlations']} highly correlated feature pairs found. "
                "Consider removing redundant features."
            )
        
        # Target imbalance
        if self.target_column and self.report["target_analysis"]:
            target_info = self.report["target_analysis"]
            if target_info.get("type") == "categorical" and not target_info.get("is_balanced"):
                recommendations.append(
                    "⚖️  Target variable is imbalanced. Consider oversampling, undersampling, or class weights."
                )
        
        if not recommendations:
            recommendations.append("✅ Data quality looks good! No major issues detected.")
        
        return recommendations


def validate_data(df, target_column=None) -> Dict[str, Any]:
    """
    Convenience function to validate data
    
    Args:
        df: DataFrame to validate
        target_column: Target column name
    
    Returns:
        Validation report dictionary
    """
    validator = DataValidator(df, target_column)
    return validator.run_full_validation()


def print_validation_report(report: Dict[str, Any]):
    """Print formatted validation report"""
    
    print("\n" + "="*60)
    print("📋 DATA VALIDATION REPORT")
    print("="*60)
    
    # Basic Info
    info = report["basic_info"]
    print(f"\n📊 Dataset Overview:")
    print(f"   Rows: {info['num_rows']:,}")
    print(f"   Columns: {info['num_columns']}")
    print(f"   Memory: {info['memory_usage_mb']:.2f} MB")
    
    # Quality Score
    print(f"\n⭐ Quality Score: {report['quality_score']:.1f}/100")
    
    # Missing Values
    missing = report["missing_values"]
    print(f"\n❓ Missing Values:")
    print(f"   Total: {missing['total_missing_cells']:,} ({missing['pct_missing_cells']:.2f}%)")
    if missing["severe_missing_cols"]:
        print(f"   Columns with >50% missing: {', '.join(missing['severe_missing_cols'])}")
    
    # Duplicates
    dup = report["duplicates"]
    if dup["has_duplicates"]:
        print(f"\n🔄 Duplicates: {dup['num_duplicates']} rows ({dup['pct_duplicates']:.2f}%)")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"   {rec}")
    
    print("\n" + "="*60)

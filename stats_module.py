"""
Statistical Analysis Module
Provides statistical tests and correlation analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr
from config import config
from logger import get_logger

logger = get_logger(__name__)


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis toolkit.
    Supports correlation, ANOVA, chi-square, t-tests, and normality tests.
    """
    
    def __init__(self, significance_level: float = None):
        self.alpha = significance_level or config.analysis.SIGNIFICANCE_LEVEL
        self.results: Dict[str, Any] = {}
    
    def analyze_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive statistical analysis on dataset.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary containing all test results
        """
        results = {
            "correlation_analysis": self.correlation_matrix(df),
            "normality_tests": self.test_normality(df),
            "outlier_summary": self.detect_outliers_summary(df)
        }
        
        # Add categorical vs numeric tests if applicable
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols and numeric_cols:
            results["anova_tests"] = self.anova_analysis(df, categorical_cols, numeric_cols)
        
        if len(categorical_cols) >= 2:
            results["chi_square_tests"] = self.chi_square_analysis(df, categorical_cols)
        
        logger.info("Statistical analysis complete")
        return results
    
    def correlation_matrix(self, df: pd.DataFrame, method: str = "pearson") -> Dict[str, Any]:
        """
        Calculate correlation matrix for numeric columns.
        
        Args:
            df: Input DataFrame
            method: Correlation method (pearson, spearman, kendall)
        
        Returns:
            Correlation matrix and significant pairs
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {"error": "Insufficient numeric columns for correlation"}
        
        try:
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr(method=method)
            
            # Find significant correlations
            significant_pairs = []
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr_val = corr_matrix.loc[col1, col2]
                    if pd.notna(corr_val) and abs(corr_val) > 0.5:  # Moderate to strong
                        significant_pairs.append({
                            "variable_1": col1,
                            "variable_2": col2,
                            "correlation": float(corr_val),
                            "strength": self._interpret_correlation(abs(corr_val)),
                            "direction": "positive" if corr_val > 0 else "negative"
                        })
            
            # Sort by absolute correlation
            significant_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            return {
                "method": method,
                "matrix": corr_matrix.to_dict(),
                "significant_correlations": significant_pairs[:10],  # Top 10
                "n_variables": len(numeric_cols)
            }
        
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {"error": str(e)}
    
    def anova_analysis(
        self, 
        df: pd.DataFrame, 
        categorical_cols: List[str], 
        numeric_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Perform one-way ANOVA tests between categorical and numeric variables.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
            numeric_cols: List of numeric column names
        
        Returns:
            List of ANOVA test results
        """
        results = []
        
        for cat_col in categorical_cols[:5]:  # Limit to avoid explosion
            for num_col in numeric_cols[:5]:
                try:
                    # Group data by categorical variable
                    groups = [
                        df[df[cat_col] == category][num_col].dropna()
                        for category in df[cat_col].unique()
                        if len(df[df[cat_col] == category][num_col].dropna()) > 0
                    ]
                    
                    # Need at least 2 groups
                    if len(groups) < 2:
                        continue
                    
                    # Perform ANOVA
                    f_stat, p_value = f_oneway(*groups)
                    
                    results.append({
                        "categorical_variable": cat_col,
                        "numeric_variable": num_col,
                        "f_statistic": float(f_stat),
                        "p_value": float(p_value),
                        "significant": p_value < self.alpha,
                        "n_groups": len(groups),
                        "interpretation": self._interpret_anova(p_value)
                    })
                
                except Exception as e:
                    logger.debug(f"ANOVA failed for {cat_col} vs {num_col}: {e}")
        
        # Sort by significance
        results.sort(key=lambda x: x["p_value"])
        return results[:10]  # Top 10 most significant
    
    def chi_square_analysis(
        self, 
        df: pd.DataFrame, 
        categorical_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Perform chi-square tests of independence between categorical variables.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
        
        Returns:
            List of chi-square test results
        """
        results = []
        
        for i, col1 in enumerate(categorical_cols[:5]):
            for col2 in categorical_cols[i+1:6]:
                try:
                    # Create contingency table
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    
                    # Perform chi-square test
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    results.append({
                        "variable_1": col1,
                        "variable_2": col2,
                        "chi_square": float(chi2),
                        "p_value": float(p_value),
                        "degrees_of_freedom": int(dof),
                        "significant": p_value < self.alpha,
                        "interpretation": self._interpret_chi_square(p_value)
                    })
                
                except Exception as e:
                    logger.debug(f"Chi-square failed for {col1} vs {col2}: {e}")
        
        # Sort by significance
        results.sort(key=lambda x: x["p_value"])
        return results[:10]
    
    def test_normality(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Test normality of numeric columns using Shapiro-Wilk test.
        
        Args:
            df: Input DataFrame
        
        Returns:
            List of normality test results
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        results = []
        
        for col in numeric_cols[:10]:  # Limit for performance
            try:
                data = df[col].dropna()
                if len(data) < 3:
                    continue
                
                # Shapiro-Wilk test (works for samples < 5000)
                if len(data) > 5000:
                    data = data.sample(5000, random_state=42)
                
                stat, p_value = stats.shapiro(data)
                
                results.append({
                    "variable": col,
                    "test_statistic": float(stat),
                    "p_value": float(p_value),
                    "is_normal": p_value > self.alpha,
                    "sample_size": len(data),
                    "interpretation": "Normal distribution" if p_value > self.alpha else "Non-normal distribution"
                })
            
            except Exception as e:
                logger.debug(f"Normality test failed for {col}: {e}")
        
        return results
    
    def detect_outliers_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect and summarize outliers across numeric columns.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Summary of outliers per column
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_summary = []
        
        for col in numeric_cols:
            try:
                data = df[col].dropna()
                if len(data) == 0:
                    continue
                
                # IQR method
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
                if len(outliers) > 0:
                    outlier_summary.append({
                        "variable": col,
                        "n_outliers": int(len(outliers)),
                        "percentage": round(len(outliers) / len(data) * 100, 2),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "min_outlier": float(outliers.min()),
                        "max_outlier": float(outliers.max())
                    })
            
            except Exception as e:
                logger.debug(f"Outlier detection failed for {col}: {e}")
        
        # Sort by percentage
        outlier_summary.sort(key=lambda x: x["percentage"], reverse=True)
        
        return {
            "method": "IQR (1.5 * IQR)",
            "columns_with_outliers": len(outlier_summary),
            "details": outlier_summary
        }
    
    def t_test_independent(
        self, 
        df: pd.DataFrame, 
        group_col: str, 
        value_col: str
    ) -> Dict[str, Any]:
        """
        Perform independent samples t-test.
        
        Args:
            df: Input DataFrame
            group_col: Column with two groups
            value_col: Numeric column to compare
        
        Returns:
            T-test results
        """
        try:
            groups = df[group_col].unique()
            if len(groups) != 2:
                return {"error": "Group column must have exactly 2 unique values"}
            
            group1_data = df[df[group_col] == groups[0]][value_col].dropna()
            group2_data = df[df[group_col] == groups[1]][value_col].dropna()
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
            
            return {
                "test": "Independent Samples T-Test",
                "group_column": group_col,
                "value_column": value_col,
                "group_1": str(groups[0]),
                "group_2": str(groups[1]),
                "group_1_mean": float(group1_data.mean()),
                "group_2_mean": float(group2_data.mean()),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < self.alpha,
                "interpretation": self._interpret_ttest(p_value, group1_data.mean(), group2_data.mean())
            }
        
        except Exception as e:
            logger.error(f"T-test failed: {e}")
            return {"error": str(e)}
    
    # Helper interpretation methods
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength"""
        if corr >= 0.7:
            return "strong"
        elif corr >= 0.4:
            return "moderate"
        elif corr >= 0.2:
            return "weak"
        else:
            return "very weak"
    
    def _interpret_anova(self, p_value: float) -> str:
        """Interpret ANOVA result"""
        if p_value < self.alpha:
            return f"Significant difference between groups (p < {self.alpha})"
        else:
            return f"No significant difference between groups (p >= {self.alpha})"
    
    def _interpret_chi_square(self, p_value: float) -> str:
        """Interpret chi-square result"""
        if p_value < self.alpha:
            return f"Variables are dependent (p < {self.alpha})"
        else:
            return f"Variables are independent (p >= {self.alpha})"
    
    def _interpret_ttest(self, p_value: float, mean1: float, mean2: float) -> str:
        """Interpret t-test result"""
        if p_value < self.alpha:
            direction = "higher" if mean1 > mean2 else "lower"
            return f"Group 1 mean is significantly {direction} than Group 2 (p < {self.alpha})"
        else:
            return f"No significant difference between groups (p >= {self.alpha})"

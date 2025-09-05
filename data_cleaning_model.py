import os
import numpy as np
import pandas as pd
import google.generativeai as genai
from typing import Tuple, Dict, Any

# Configure Gemini key (prefer ENV)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not set. Please add it to your .env file.")
genai.configure(api_key=api_key)

class DataCleaningModel:
    """
    Minimal, dataset-agnostic cleaner:
    - Fill missing values per-column (numeric: median, categorical: mode)
    - Standardize *any* column containing 'date' in its name to YYYY-MM-DD
    - Optional winsorize numeric columns (IQR method)
    - NO domain rules (e.g., no District/State assumptions)
    """
    def __init__(self, winsorize: bool = True, use_llm_for_typos: bool = False):
        self.winsorize = winsorize
        self.use_llm_for_typos = use_llm_for_typos
        self.report: Dict[str, Any] = {}

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        self.report = {
            "missing_values_filled": {},
            "date_columns_standardized": [],
            "outliers_capped": {},
        }

        df = self._fill_missing(df)
        df = self._standardize_dates(df)
        if self.winsorize:
            df = self._winsorize_numeric(df)

        return df, self.report

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            miss_idx = df[df[col].isna()].index.tolist()
            if not miss_idx:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
            else:
                mode_series = df[col].mode()
                fill_value = mode_series.iloc[0] if not mode_series.empty else ""

            df[col] = df[col].fillna(fill_value)
            self.report["missing_values_filled"][col] = {
                "count": len(miss_idx),
                "value_used": fill_value if not isinstance(fill_value, (np.floating, np.integer)) else float(fill_value),
            }
        return df

    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        date_cols = [c for c in df.columns if "date" in c.lower()]
        for col in date_cols:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                df[col] = parsed.dt.strftime("%Y-%m-%d")
                self.report["date_columns_standardized"].append(col)
            except Exception:
                # leave as-is if parsing fails entirely
                pass
        return df

    def _winsorize_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            mask_lower = df[col] < lower
            mask_upper = df[col] > upper
            if mask_lower.any() or mask_upper.any():
                before = df.loc[mask_lower | mask_upper, col].tolist()
                df.loc[mask_lower, col] = lower
                df.loc[mask_upper, col] = upper
                after = df.loc[mask_lower | mask_upper, col].tolist()
                self.report["outliers_capped"][col] = {
                    "count": int(mask_lower.sum() + mask_upper.sum()),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "examples_before_after": list(zip(before[:5], after[:5])),
                }
        return df
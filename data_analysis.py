import os
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import google.generativeai as genai
from json_encoder import to_py

# =======================
# Gemini config
# =======================
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not set. Please add it to your .env file.")

genai.configure(api_key=api_key)
gemini = genai.GenerativeModel("gemini-2.5-pro")

# =======================
# Helpers: summaries
# =======================
def numeric_summary(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in df.select_dtypes(include=np.number).columns:
        desc = df[col].describe().to_dict()
        q1, q3 = desc.get("25%", np.nan), desc.get("75%", np.nan)
        if pd.notna(q1) and pd.notna(q3):
            iqr = q3 - q1
            lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            lb = ub = np.nan
        desc["missing"] = int(df[col].isnull().sum())
        desc["outliers"] = {
            "lower_bound": float(lb) if pd.notna(lb) else None,
            "upper_bound": float(ub) if pd.notna(ub) else None,
        }
        out[col] = to_py(desc)
    return out

def categorical_summary(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in df.select_dtypes(include="object").columns:
        try:
            safe_series = df[col].astype(str)
            out[col] = {
                "missing": int(safe_series.isnull().sum()),
                "unique_count": int(safe_series.nunique()),
                "top_5_values": to_py(safe_series.value_counts().head(5).to_dict()),
            }
        except Exception as e:
            out[col] = {"error": f"Could not summarize column: {str(e)}"}
    return out

# =======================
# Helpers: detection
# =======================
def _detect_date_columns(df: pd.DataFrame) -> List[str]:
    date_cols: List[str] = list(df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns)
    for c in df.select_dtypes(include="object").columns:
        sample = df[c].dropna().head(20)
        try:
            pd.to_datetime(sample, errors="raise")
            date_cols.append(c)
        except Exception:
            continue
    return list(dict.fromkeys(date_cols))

def _coerce_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    if not date_cols:
        return df
    df2 = df.copy()
    for c in date_cols:
        try:
            df2[c] = pd.to_datetime(df2[c], errors="coerce")
        except Exception:
            pass
    return df2

# =======================
# KPIs
# =======================
def generate_kpis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    kpis = [
        {"name": "Total Rows", "value": int(df.shape[0])},
        {"name": "Total Columns", "value": int(df.shape[1])},
        {"name": "Missing %", "value": round((df.isnull().sum().sum() / (df.shape[0]*df.shape[1])) * 100, 2)},
    ]
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols[:3]:
        s = pd.to_numeric(df[col], errors="coerce")
        kpis.append({"name": f"Average {col}", "value": float(s.mean())})
        kpis.append({"name": f"Best {col}", "value": float(s.max())})
        kpis.append({"name": f"Worst {col}", "value": float(s.min())})
    dcols = _detect_date_columns(df)
    if dcols:
        dcol = dcols[0]
        rng = _coerce_dates(df, [dcol])[dcol].dropna()
        if not rng.empty:
            kpis.append({"name": f"{dcol} Range", "value": f"{rng.min().date()} → {rng.max().date()}"})
    return kpis

# =======================
# Charts
# =======================
def _sanitize(obj):
    """
    Recursively sanitize Plotly JSON:
    - Convert "True"/"False" -> Python bool
    - Convert numeric strings -> float/int
    - Fix domain arrays (pie/bar/heatmap ranges)
    - Fix colorscales (convert first element to float)
    """
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            if k == "error":
                continue
            clean[k] = _sanitize(v)
        return clean

    elif isinstance(obj, list):
        # Detect colorscale: list of 2-element lists
        if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in obj):
            fixed = []
            for a, b in obj:
                # ensure first element is float
                try:
                    a = float(a) if isinstance(a, str) else a
                except Exception:
                    pass
                fixed.append([a, b])
            return fixed

        # Otherwise sanitize list elements
        return [_sanitize(v) for v in obj]

    elif isinstance(obj, str):
        lower = obj.lower()

        # Boolean fix
        if lower == "false":
            return False
        if lower == "true":
            return True

        # Number fix
        try:
            if "." in obj:
                return float(obj)
            return int(obj)
        except Exception:
            return obj

    else:
        return obj


def plotly_to_safe_dict(fig) -> Dict[str, Any]:
    d = fig.to_plotly_json()
    d["layout"].pop("template", None)
    return _sanitize(d)

def llm_chart_suggestions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []

    num_cols = [c for c in df.select_dtypes(include=np.number).columns if "id" not in c.lower()]
    cat_cols = [c for c in df.select_dtypes(include="object").columns if "id" not in c.lower()]
    date_cols = _detect_date_columns(df)

    candidates = []

    # Ensure categorical diversity
    for i, c in enumerate(cat_cols[:5]):  # up to 5 categorical
        if num_cols:
            candidates.append({"chart_type": "bar", "columns": [c, num_cols[i % len(num_cols)]]})
            candidates.append({"chart_type": "box", "columns": [c, num_cols[i % len(num_cols)]]})
        if i < len(num_cols):
            candidates.append({"chart_type": "pie", "columns": [c, num_cols[i]]})

    # Always include key numeric visualizations
    for n in num_cols[:5]:
        candidates.append({"chart_type": "histogram", "columns": [n]})
        candidates.append({"chart_type": "density", "columns": [n]})

    # Multi-numeric comparisons
    if len(num_cols) >= 2:
        candidates.append({"chart_type": "scatter", "columns": num_cols[:2]})
        candidates.append({"chart_type": "trendline_scatter", "columns": num_cols[:2]})
    if len(num_cols) >= 3:
        candidates.append({"chart_type": "bubble", "columns": num_cols[:3]})
        candidates.append({"chart_type": "correlation_matrix", "columns": num_cols})

    # Time-series
    if date_cols and num_cols:
        candidates.append({"chart_type": "line", "columns": [date_cols[0], num_cols[0]]})
        candidates.append({"chart_type": "area", "columns": [date_cols[0], num_cols[0]]})

    # Pick 6–7 diverse charts
    seen, diverse = set(), []
    for cand in candidates:
        key = (cand["chart_type"], tuple(cand["columns"]))
        if key not in seen:
            seen.add(key)
            diverse.append(cand)
        if len(diverse) >= 7:
            break

    return diverse

    
def summarize_chart(chart_type: str, cols: List[str], df: pd.DataFrame) -> str:
    if not cols or any(c not in df for c in cols):
        return f"{chart_type.title()} chart with {cols}."
    try:
        sample = df[cols].dropna().head(15).to_dict("records")
        prompt = f"""
        Chart type: {chart_type}, Columns: {cols}.
        Sample: {sample}.
        Write 1-2 professional business insights.
        """
        resp = gemini.generate_content(prompt)
        if resp.text and len(resp.text) > 20:
            return resp.text.strip()
    except:
        pass
    if chart_type in ["bar", "pie"] and len(cols) >= 2:
        vc = df.groupby(cols[0])[cols[1]].mean().sort_values()
        return f"{cols[0]} groups range from {vc.min():.2f} ({vc.idxmin()}) to {vc.max():.2f} ({vc.idxmax()})."
    elif chart_type == "histogram":
        s = pd.to_numeric(df[cols[0]], errors="coerce").dropna()
        return f"{cols[0]} ranges {s.min():.1f}–{s.max():.1f}, average {s.mean():.1f}."
    return f"{chart_type.title()} chart with {cols}."

def build_charts(df: pd.DataFrame, chart_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    charts = {}
    df = _coerce_dates(df, _detect_date_columns(df))
    for spec in chart_specs:
        chart_type, cols = spec.get("chart_type", "").lower(), spec.get("columns", [])
        try:
            fig = None
            if chart_type == "bar" and len(cols) >= 2:
                fig = px.bar(df, x=cols[0], y=cols[1])
            elif chart_type == "line" and len(cols) >= 2:
                fig = px.line(df, x=cols[0], y=cols[1])
            elif chart_type == "scatter" and len(cols) >= 2:
                fig = px.scatter(df, x=cols[0], y=cols[1])
            elif chart_type == "trendline_scatter" and len(cols) >= 2:
                fig = px.scatter(df, x=cols[0], y=cols[1], trendline="ols")
            elif chart_type == "histogram" and cols:
                fig = px.histogram(df, x=cols[0])
            elif chart_type == "pie" and len(cols) >= 2:
                fig = px.pie(df, names=cols[0], values=cols[1])
            elif chart_type == "area" and len(cols) >= 2:
                fig = px.area(df, x=cols[0], y=cols[1])
            elif chart_type == "bubble" and len(cols) >= 3:
                fig = px.scatter(df, x=cols[0], y=cols[1], size=cols[2], color=cols[0])
            elif chart_type == "heatmap" and len(cols) >= 2:
                fig = px.density_heatmap(df, x=cols[0], y=cols[1])
            elif chart_type == "box" and len(cols) >= 2:
                fig = px.box(df, x=cols[0], y=cols[1])
            elif chart_type == "violin" and cols:
                fig = px.violin(df, y=cols[0])
            elif chart_type == "density" and cols:
                fig = px.histogram(df, x=cols[0], histnorm="density")
            elif chart_type == "correlation_matrix" and len(cols) >= 2:
                corr = df[cols].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")

            if fig:
                charts[chart_type] = {
                    "figure": plotly_to_safe_dict(fig),
                    "summary": summarize_chart(chart_type, cols, df)
                }
        except Exception as e:
            charts[chart_type] = {"error": str(e), "summary": f"{chart_type.title()} chart error"}
    return charts
import random

# =======================
# Random Chart Suggestions
# =======================
def random_chart_suggestions(df: pd.DataFrame, max_charts: int = 6) -> List[Dict[str, Any]]:
    if df.empty:
        return []

    num_cols = [c for c in df.select_dtypes(include=np.number).columns if "id" not in c.lower()]
    cat_cols = [c for c in df.select_dtypes(include="object").columns if "id" not in c.lower()]
    date_cols = _detect_date_columns(df)

    candidates = []

    # Cat + Num charts
    for c in cat_cols:
        for n in num_cols:
            candidates += [
                {"chart_type": "bar", "columns": [c, n]},
                {"chart_type": "box", "columns": [c, n]},
                {"chart_type": "pie", "columns": [c, n]},
            ]

    # Numeric only
    for n in num_cols:
        candidates += [
            {"chart_type": "histogram", "columns": [n]},
            {"chart_type": "density", "columns": [n]},
        ]

    # Num vs Num
    if len(num_cols) >= 2:
        candidates.append({"chart_type": "scatter", "columns": num_cols[:2]})
        if len(num_cols) >= 3:
            candidates.append({"chart_type": "bubble", "columns": num_cols[:3]})

    # Date vs Num
    if date_cols and num_cols:
        candidates += [
            {"chart_type": "line", "columns": [date_cols[0], num_cols[0]]},
            {"chart_type": "area", "columns": [date_cols[0], num_cols[0]]},
        ]

    # Shuffle and pick random charts
    random.shuffle(candidates)
    selected = []
    seen = set()
    for cand in candidates:
        key = (cand["chart_type"], tuple(cand["columns"]))
        if key not in seen:
            seen.add(key)
            selected.append(cand)
        if len(selected) >= max_charts:
            break

    return selected
def generate_charts(df: pd.DataFrame) -> Dict[str, Any]:
    recs = random_chart_suggestions(df, max_charts=6)  # <-- RANDOMIZER HERE
    uniq, seen = [], set()
    for r in recs:
        key = (r["chart_type"], tuple(r["columns"]))
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    # Add a heatmap for numeric correlation (if possible)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) > 2:
        uniq.append({"chart_type": "heatmap", "columns": [num_cols[0], num_cols[1]]})

    return build_charts(df, uniq[:8])


# =======================
# Data Story
# =======================
def data_story(df: pd.DataFrame, report: dict) -> str:
    sample = df.head(5).to_dict("records")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = _detect_date_columns(df)

    prompt = f"""
    You are a senior business data analyst.
    Dataset: {df.shape[0]} rows × {df.shape[1]} cols.
    Numeric: {num_cols}
    Categorical: {cat_cols}
    Dates: {date_cols}
    Sample: {json.dumps(to_py(sample))}
    Write a 2–3 paragraph business-style summary:
    - Describe dataset coverage and possible domain.
    - Report averages, best, worst in key numeric columns.
    - Mention category distributions & anomalies.
    - Highlight risks, strengths, and potential opportunities.
    """
    try:
        resp = gemini.generate_content(prompt, generation_config={"temperature": 0.6})
        if resp.text and len(resp.text) > 50:
            return resp.text.strip()
    except:
        pass

    parts = [f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns."]
    missing = int(df.isnull().sum().sum())
    parts.append("No missing values." if missing == 0 else f"{missing} missing values detected.")
    if date_cols:
        rng = _coerce_dates(df, date_cols)[date_cols[0]].dropna()
        if not rng.empty:
            parts.append(f"Date range spans {rng.min().date()} to {rng.max().date()}.")
    for col in num_cols[:3]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.empty:
            parts.append(f"In {col}, avg={s.mean():.2f}, min={s.min():.2f}, max={s.max():.2f}.")
    for col in cat_cols[:2]:
        vc = df[col].value_counts()
        if not vc.empty:
            parts.append(f"For {col}, most common='{vc.idxmax()}' ({vc.max()}) vs least='{vc.idxmin()}' ({vc.min()}).")
    parts.append("Overall, the dataset shows key trends and variability useful for decision-making.")
    return " ".join(parts)

# =======================
# Slicers & Filters
# =======================

def generate_slicers(df: pd.DataFrame) -> Dict[str, Any]:
    slicers = {}

    # categorical slicers → dropdown/multiselect
    for col in df.select_dtypes(include="object").columns:
        unique_vals = df[col].dropna().unique().tolist()
        if 2 <= len(unique_vals) <= 50:
            slicers[col] = {
                "type": "categorical",
                "options": unique_vals
            }

    # numeric slicers → slider
    for col in df.select_dtypes(include=np.number).columns:
        rng = df[col].dropna()
        if not rng.empty:
            slicers[col] = {
                "type": "numeric",
                "min": float(rng.min()),
                "max": float(rng.max())
            }

    # date slicers → date picker
    for col in _detect_date_columns(df):
        rng = pd.to_datetime(df[col], errors="coerce").dropna()
        if not rng.empty:
            slicers[col] = {
                "type": "date",
                "min": rng.min().isoformat(),
                "max": rng.max().isoformat()
            }

    return slicers


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if not filters:
        return df
    df2 = df.copy()
    for col, condition in filters.items():
        if col not in df2:
            continue
        try:
            if not isinstance(condition, (list, dict)):
                df2 = df2[df2[col] == condition]
            elif isinstance(condition, list):
                df2 = df2[df2[col].isin(condition)]
            elif isinstance(condition, dict):
                if "min" in condition:
                    df2 = df2[df2[col] >= condition["min"]]
                if "max" in condition:
                    df2 = df2[df2[col] <= condition["max"]]
        except Exception:
            continue
    return df2

# =======================
# Main
# =======================
def analyze_data(df: pd.DataFrame, cleaning_report: dict, filters: dict = None) -> Dict[str, Any]:
    df_filtered = apply_filters(df, filters)

    return to_py({
        "shape": {"rows": int(df_filtered.shape[0]), "columns": int(df_filtered.shape[1])},
        "missing_values": to_py(df_filtered.isnull().sum().to_dict()),
        "numeric_summary": numeric_summary(df_filtered),
        "categorical_summary": categorical_summary(df_filtered),
        "kpis": generate_kpis(df_filtered),
        "charts": generate_charts(df_filtered),       # 🔀 RANDOMIZED charts already handled inside
        "data_story": data_story(df_filtered, cleaning_report),
        "slicers": generate_slicers(df),              # ✅ always from full dataset
        "active_filters": filters or {}               # ✅ frontend sees what’s applied
    })


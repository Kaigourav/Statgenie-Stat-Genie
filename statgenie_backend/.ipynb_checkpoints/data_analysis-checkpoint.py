import os
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import google.generativeai as genai
from json_encoder import to_py

# Gemini config
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = "AIzaSyCNMBV3murjNCR61TKTlzr28vwCq-kuUS8"
genai.configure(api_key=api_key)
gemini = genai.GenerativeModel('gemini-1.5-flash')

# ----- helpers -----

def numeric_summary(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in df.select_dtypes(include=np.number).columns:
        desc = df[col].describe().to_dict()
        q1, q3 = desc['25%'], desc['75%']
        iqr = q3 - q1
        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr
        desc['missing'] = int(df[col].isnull().sum())
        desc['outliers'] = {"lower_bound": float(lb), "upper_bound": float(ub)}
        out[col] = to_py(desc)
    return out

def categorical_summary(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in df.select_dtypes(include='object').columns:
        out[col] = {
            "missing": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
            "top_5_values": to_py(df[col].value_counts().head(5).to_dict()),
        }
    return out

def kpis(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # Look for common KPI column names
    for potential_col, kpi_name in [('Population', 'total_population'), 
                                   ('Revenue', 'total_revenue'),
                                   ('Sales', 'total_sales'),
                                   ('Price', 'average_price')]:
        if potential_col in df.columns and pd.api.types.is_numeric_dtype(df[potential_col]):
            out[kpi_name] = float(df[potential_col].sum() if 'total' in kpi_name else df[potential_col].mean())
    
    # Generic fallback: use first numeric column
    if not out and len(numeric_cols) > 0:
        out['average_value'] = float(df[numeric_cols[0]].mean())
        
    return out

def llm_chart_suggestions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
        
    sample = df.head(1).to_dict('records')[0]
    prompt = (
        "You are a data visualization expert. Suggest 3-4 charts.\n"
        "Return pure JSON array of objects: "
        '[{"chart_type":"bar","columns":["col"]},{"chart_type":"scatter","columns":["x","y"]}, ...]\n'
        f"Sample row:\n{json.dumps(to_py(sample), ensure_ascii=False)}"
    )
    try:
        resp = gemini.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(resp.text)
    except Exception:
        # safe defaults
        defaults = []
        cat_cols = df.select_dtypes(include='object').columns
        num_cols = df.select_dtypes(include=np.number).columns
        
        if len(cat_cols) > 0:
            defaults.append({"chart_type": "bar", "columns": [cat_cols[0]]})
        if len(num_cols) >= 2:
            defaults.append({"chart_type": "scatter", "columns": [num_cols[0], num_cols[1]]})
        if len(cat_cols) > 0 and len(num_cols) > 0:
            defaults.append({"chart_type": "box", "columns": [cat_cols[0], num_cols[0]]})
            
        return defaults
        
def plotly_to_safe_dict(fig) -> Dict[str, Any]:
    """Guarantee JSON-safe Plotly spec using plotly.io.to_json."""
    fig_json = pio.to_json(fig, pretty=False)
    return json.loads(fig_json)

def build_charts(df: pd.DataFrame, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    charts: Dict[str, Any] = {}
    for rec in recs or []:
        if not rec or 'chart_type' not in rec:
            continue
            
        chart_type = rec.get('chart_type')
        cols = rec.get('columns', [])
        
        try:
            if chart_type == 'bar' and len(cols) == 1 and cols[0] in df.columns:
                c = cols[0]
                counts = df[c].value_counts().head(10).reset_index()
                counts.columns = [c, 'Count']
                fig = px.bar(counts, x=c, y='Count', title=f'Top 10 {c}')
                charts[f'{c}_bar'] = plotly_to_safe_dict(fig)

            elif chart_type == 'scatter' and len(cols) == 2 and all(col in df.columns for col in cols):
                fig = px.scatter(df, x=cols[0], y=cols[1], title=f'{cols[0]} vs {cols[1]}')
                charts[f'scatter_{cols[0]}_{cols[1]}'] = plotly_to_safe_dict(fig)

            elif chart_type == 'pie' and len(cols) == 1 and cols[0] in df.columns:
                c = cols[0]
                vc = df[c].value_counts().head(10)  # Limit to top 10 for pie charts
                fig = px.pie(values=vc.values, names=vc.index, title=f'{c} Distribution (Top 10)')
                charts[f'{c}_pie'] = plotly_to_safe_dict(fig)
                
            elif chart_type == 'box' and len(cols) == 2 and all(col in df.columns for col in cols):
                # Assuming first col is categorical, second is numeric
                fig = px.box(df, x=cols[0], y=cols[1], title=f'{cols[1]} by {cols[0]}')
                charts[f'box_{cols[0]}_{cols[1]}'] = plotly_to_safe_dict(fig)
                
        except Exception as e:
            # Log chart building errors but don't break the entire process
            continue
            
    return charts

def data_story(df: pd.DataFrame, report: dict) -> str:
    """
    Generates a comprehensive, business-friendly data story from a DataFrame.

    This function now includes a detailed analysis of best and worst performers
    and key averages, in addition to the standard data profile.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        report (dict): A dictionary containing a data cleaning report.

    Returns:
        str: A generated text story based on the data analysis.
    """
    # 1. Identify a key numeric column for performance analysis.
    # This logic assumes we want to analyze the first numeric column found.
    # You can customize this to select a specific column (e.g., 'revenue', 'sales').
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        kpi_column = numeric_cols[0]
        
        # 2. Calculate average, best, and worst performers based on this column.
        average_value = df[kpi_column].mean()
        
        # Find the row with the best performance.
        best_performer_row = df.loc[df[kpi_column].idxmax()]
        best_performer_data = {
            "value": best_performer_row[kpi_column],
            "details": best_performer_row.to_dict()
        }
        
        # Find the row with the worst performance.
        worst_performer_row = df.loc[df[kpi_column].idxmin()]
        worst_performer_data = {
            "value": worst_performer_row[kpi_column],
            "details": worst_performer_row.to_dict()
        }

        # 3. Add a new performance summary to the main summary dictionary.
        performance_summary = {
            "analyzed_metric": kpi_column,
            "average": average_value,
            "best_performer": best_performer_data,
            "worst_performer": worst_performer_data
        }
    else:
        performance_summary = None

    # 4. Construct the complete summary for the model.
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "cleaning_report": report,
        "numeric_summary": numeric_summary(df),
        "categorical_summary": categorical_summary(df),
        "performance_summary": performance_summary
    }

    # 5. Update the prompt to ask for specific insights from the new data.
    prompt = (
        "Write a concise, business-friendly data story based on the following JSON. "
        "The story should include an overview of the dataset, key findings from the "
        "data cleaning report, and a specific analysis of the best and worst performers "
        "and the average value for the key metric identified in the performance_summary. "
        "Highlight any interesting details about these performers.\n"
        f"{json.dumps(to_py(summary), ensure_ascii=False)}"
    )

    try:
        # Assumes 'gemini' is a working model client.
        # resp = gemini.generate_content(prompt)
        # return resp.text
        return f"Mock story generated based on prompt: {prompt}"
    except Exception as e:
        return f"Story generation failed: {e}"


# ----- main entry -----
def analyze_data(df: pd.DataFrame, cleaning_report: dict) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "missing_values": to_py(df.isnull().sum().to_dict()),
        "numeric_summary": numeric_summary(df),
        "categorical_summary": categorical_summary(df),
        "kpis": kpis(df),
        "charts": build_charts(df, llm_chart_suggestions(df)),
        "data_story": data_story(df, cleaning_report)
    }
    # Strong guarantee that we only return pure-Python types
    return to_py(results)
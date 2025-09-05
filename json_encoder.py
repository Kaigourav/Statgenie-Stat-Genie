# json_encoder.py
import json
from typing import Any
import numpy as np
import pandas as pd
from flask import Response

def to_py(o: Any):
    """
    Recursively convert numpy/pandas/plotly objects to plain Python.
    Ensures JSON-safe output with correct types (no NaN/Inf).
    """
    try:
        # --- primitive passthrough ---
        if isinstance(o, (bool, int, float, type(None))):
            # replace NaN/Inf with None
            if isinstance(o, float):
                if np.isnan(o) or np.isinf(o):
                    return None
            return o

        if isinstance(o, np.generic):
            val = o.item()
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return None
            return val

        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if isinstance(o, pd.Timedelta):
            return o.total_seconds()
        if isinstance(o, pd.Series):
            return [to_py(v) for v in o.tolist()]
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        if isinstance(o, np.ndarray):
            return [to_py(v) for v in o.tolist()]
        if hasattr(o, "to_plotly_json"):
            return to_py(o.to_plotly_json())
        if isinstance(o, dict):
            return {str(to_py(k)): to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, set)):
            return [to_py(v) for v in o]

        # --- fallback ---
        return str(o)

    except Exception:
        return f"<Unserializable: {type(o).__name__}>"



def safe_json_response(payload: Any, status: int = 200) -> Response:
    """
    Safe HTTP JSON response (orjson if available, else json).
    """
    try:
        import orjson
        body = orjson.dumps(payload, default=to_py)
    except (ImportError, TypeError):
        try:
            body = json.dumps(payload, default=to_py, ensure_ascii=False)
        except TypeError:
            body = json.dumps(to_py(payload), ensure_ascii=False)
    return Response(body, status=status, mimetype="application/json")

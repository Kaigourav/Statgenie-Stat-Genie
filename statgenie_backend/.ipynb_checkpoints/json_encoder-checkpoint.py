import json
from typing import Any
import numpy as np
import pandas as pd
from flask import Response

def to_py(o: Any):
    """
    Recursively convert numpy/pandas/plotly-friendly objects to plain Python.
    Includes a final fallback to string representation for truly unknown types.
    """
    try:
        # Scalars (int64, float64, etc.)
        if isinstance(o, np.generic):
            return o.item()
        # Pandas Timestamps
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        # Pandas Timedeltas
        if isinstance(o, pd.Timedelta):
            return o.total_seconds()
        # Pandas Series
        if isinstance(o, pd.Series):
            return [to_py(v) for v in o.tolist()]
        # Pandas DataFrames
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        # Numpy arrays
        if isinstance(o, np.ndarray):
            return [to_py(v) for v in o.tolist()]
        # Dictionaries
        if isinstance(o, dict):
            return {str(to_py(k)): to_py(v) for k, v in o.items()}
        # Lists, Tuples, Sets
        if isinstance(o, (list, tuple, set)):
            return [to_py(v) for v in o]
        # Fallback: try to get a string representation as a last resort
        else:
            return str(o)
    except Exception as e:
        # If even str() fails, return an error string.
        return f"<Unserializable: {type(o).__name__}>"

def safe_json_response(payload: Any, status: int = 200) -> Response:
    """
    Safe HTTP JSON response that never throws on NumPy/Pandas/Datetime/etc.
    Uses orjson for performance if available, falls back to standard json.
    """
    try:
        # Try using the faster orjson first
        import orjson
        body = orjson.dumps(payload, default=to_py)
    except (ImportError, TypeError):
        # Fallback to standard json if orjson is not installed or fails
        try:
            body = json.dumps(payload, default=to_py, ensure_ascii=False)
        except TypeError:
            # Final fallback: force conversion of the entire payload
            body = json.dumps(to_py(payload), ensure_ascii=False)
    return Response(body, status=status, mimetype='application/json')
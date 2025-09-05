import pandas as pd
import json

def read_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return pd.DataFrame.from_dict(data)
    elif isinstance(data, list):
        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported JSON structure")

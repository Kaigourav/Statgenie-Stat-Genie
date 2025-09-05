import pandas as pd

import pandas as pd

def read_csv_excel(file_path: str) -> pd.DataFrame:
    """
    Load CSV or Excel files properly into a DataFrame.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

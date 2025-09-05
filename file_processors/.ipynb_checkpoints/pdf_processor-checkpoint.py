import pandas as pd
import camelot
import pdfplumber
from .unstructured_parser import ai_extract_table_to_csv

def read_pdf(path: str, csv_output_path: str = None) -> pd.DataFrame:
    """
    Extract tables from PDF, fallback to AI for unstructured text.
    """
    tables = []
    try:
        camelot_tables = camelot.read_pdf(path, pages="all", flavor="stream")
        for t in camelot_tables:
            tables.append(t.df)
    except Exception:
        pass

    # Combine all Camelot tables if found
    if tables:
        df = pd.concat(tables, ignore_index=True)
        if csv_output_path:
            df.to_csv(csv_output_path, index=False)
        return df

    # Fallback: pdfplumber text extraction
    text_lines = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_lines.extend(text.splitlines())
    except Exception:
        pass

    # AI-assisted extraction if text found
    if text_lines:
        if csv_output_path:
            return ai_extract_table_to_csv(text_lines, csv_output_path)
        return ai_extract_table_to_csv(text_lines, None)

    # If nothing found
    return pd.DataFrame()

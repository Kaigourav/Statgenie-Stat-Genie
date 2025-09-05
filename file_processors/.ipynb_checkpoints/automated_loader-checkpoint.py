import os
import pandas as pd
from .csv_excel import read_csv_excel
from .pdf_processor import read_pdf
from .text_processor import read_txt
from .docx_processor import read_docx
from .json_processor import read_json
from .image_processor import read_image
from .unstructured_parser import ai_extract_table_to_csv

def automated_load(file_path: str, filename: str, csv_output_path: str) -> pd.DataFrame:
    """
    Fully automated loader: detects type, extracts structured table, saves CSV.
    """
    ext = filename.lower()

    df = pd.DataFrame()

    try:
        if ext.endswith((".csv", ".xls", ".xlsx")):
            df = read_csv_excel(file_path)
            df.to_csv(csv_output_path, index=False)

        elif ext.endswith(".pdf"):
            df = read_pdf(file_path)
            if "text" in df.columns:
                df = ai_extract_table_to_csv(df["text"].tolist(), csv_output_path)
            else:
                df.to_csv(csv_output_path, index=False)

        elif ext.endswith(".txt"):
            df = read_txt(file_path)
            df = ai_extract_table_to_csv(df["text"].tolist(), csv_output_path)

        elif ext.endswith(".docx"):
            df = read_docx(file_path)
            if "text" in df.columns:
                df = ai_extract_table_to_csv(df["text"].tolist(), csv_output_path)
            else:
                df.to_csv(csv_output_path, index=False)

        elif ext.endswith(".json"):
            df = read_json(file_path)
            df.to_csv(csv_output_path, index=False)

        elif ext.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
            df = read_image(file_path)
            df = ai_extract_table_to_csv(df["text"].tolist(), csv_output_path)

        else:
            raise ValueError(f"Unsupported file type: {filename}")

    except Exception as e:
        print(f"[ERROR] Failed to load {filename}: {e}")
        # Fallback: save raw text CSV
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            df = ai_extract_table_to_csv(lines, csv_output_path)

    return df

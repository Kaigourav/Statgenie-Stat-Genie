import pandas as pd
import json
from typing import List
import google.generativeai as genai
import os

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
gemini = genai.GenerativeModel("gemini-1.5-flash")

def ai_extract_table_to_csv(text_lines: List[str], output_csv_path: str) -> pd.DataFrame:
    """
    Convert unstructured text lines into structured table via LLM, then save as CSV.
    """
    if not text_lines:
        return pd.DataFrame()

    prompt = f"""
You are a data extraction assistant. Extract structured tables from text.
Keep rows and columns intact. Output ONLY JSON array of objects.

Text sample:
{text_lines[:2000]}
"""

    try:
        resp = gemini.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        data = json.loads(resp.text)
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            df.to_csv(output_csv_path, index=False)
            return df
    except Exception as e:
        print("LLM parsing failed:", e)

    # Fallback: save as single-column CSV
    df = pd.DataFrame({"text": text_lines})
    df.to_csv(output_csv_path, index=False)
    return df

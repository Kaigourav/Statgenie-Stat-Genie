import os
import io
import pandas as pd
import pdfplumber
import docx
import google.generativeai as genai

# --- Gemini config ---
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
gemini = genai.GenerativeModel("gemini-2.0-flash")


def extract_text(file_path: str) -> str:
    """Extract raw text from unstructured files."""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type for unstructured parsing: {file_path}")


def llm_to_csv(text: str) -> pd.DataFrame:
    """Ask Gemini to turn free-form text into a CSV table."""
    prompt = f"""
You are a data parser.
Convert the following unstructured document into a clean CSV table with rows and columns.
Ensure consistent headers and data rows. Output ONLY raw CSV, no explanation.

Document:
{text[:5000]}   # truncated to 5000 chars
"""
    resp = gemini.generate_content(prompt)
    csv_str = resp.text.strip().strip("```csv").strip("```")
    return pd.read_csv(io.StringIO(csv_str))


def process_unstructured(file_path: str) -> pd.DataFrame:
    """Full pipeline: extract → LLM → DataFrame"""
    raw_text = extract_text(file_path)
    df = llm_to_csv(raw_text)
    return df

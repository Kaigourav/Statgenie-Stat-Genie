import pandas as pd
import google.generativeai as genai
import os
from io import StringIO   

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
gemini = genai.GenerativeModel("gemini-2.0-flash")

def process_image(file_path: str) -> pd.DataFrame:
    """
    Let LLM handle OCR + structuring: 
    Input = image file, Output = structured CSV → DataFrame.
    """
    try:
        # Ask Gemini directly with image
        prompt = """
        You are a data extraction expert.
        The user provided an image that may contain a table or structured data.
        Your job:
        - Read text from the image
        - Convert it into a clean CSV table
        - Ensure the first row contains headers
        - Output ONLY the CSV (no explanations, no markdown fences)
        """

        resp = gemini.generate_content(
            [prompt, {"mime_type": "image/png", "data": open(file_path, "rb").read()}]
        )

        # Extract CSV text from Gemini
        csv_text = resp.text.strip().strip("```csv").strip("```").strip()

        # Convert to DataFrame
        df = pd.read_csv(StringIO(csv_text))
        return df

    except Exception as e:
        print(f"[ERROR] Image processing failed: {e}")
        return pd.DataFrame()

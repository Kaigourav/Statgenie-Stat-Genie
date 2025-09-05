import pandas as pd
import pytesseract
from PIL import Image
import cv2
import numpy as np
from .unstructured_parser import ai_extract_table_to_csv

def read_image(path: str, csv_output_path: str = None) -> pd.DataFrame:
    """
    Extract structured table from image using:
    1. Table detection (OpenCV)
    2. OCR (pytesseract)
    3. AI fallback for unstructured content
    """
    # Load image
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 4)

    # Detect contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_lines = []

    for c in sorted(contours, key=lambda x: cv2.boundingRect(x)[1]):  # top to bottom
        x, y, w, h = cv2.boundingRect(c)
        roi = img[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi)
        if text.strip():
            text_lines.extend(text.splitlines())

    if not text_lines:  # fallback full OCR
        text = pytesseract.image_to_string(Image.open(path))
        text_lines = [line.strip() for line in text.splitlines() if line.strip()]

    # AI-assisted table reconstruction
    if csv_output_path:
        return ai_extract_table_to_csv(text_lines, csv_output_path)
    return ai_extract_table_to_csv(text_lines, None)

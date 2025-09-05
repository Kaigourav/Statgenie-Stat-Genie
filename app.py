import os
import traceback
import json
import uuid
import pandas as pd
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import plotly.io as pio
import kaleido

from data_cleaning_model import DataCleaningModel
from data_analysis import analyze_data
from json_encoder import safe_json_response
from file_processors.automated_loader import automated_load
from unstructured_handler import process_unstructured
from image_handler import process_image
from pdf_report import generate_pdf_report

from dotenv import load_dotenv
load_dotenv() 

# -------------------------------
# Job storage (in-memory)
# -------------------------------
jobs = {}  

# -------------------------------
# Flask App Setup
# -------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = (
    ".csv", ".xls", ".xlsx", ".json",
    ".txt", ".pdf", ".doc", ".docx",
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff"
)

# -------------------------------
# Load DataFrame with modular processors
# -------------------------------
def load_dataframe(file_path: str, filename: str) -> pd.DataFrame:
    ext = os.path.splitext(filename)[1].lower()

    if ext in [".csv", ".xls", ".xlsx", ".json"]:
        csv_output = os.path.splitext(file_path)[0] + "_structured.csv"
        return automated_load(file_path, filename, csv_output)

    elif ext in [".pdf", ".doc", ".docx", ".txt"]:
        return process_unstructured(file_path)

    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        return process_image(file_path)

    else:
        raise ValueError(f"Unsupported file type for loading: {ext}")

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    return safe_json_response({"msg": "StatGenie API live"})

@app.route("/health")
def health():
    return safe_json_response({"status": "ok"})

@app.route("/upload_page")
def upload_page():
    return render_template("uploader.html")

@app.route("/clean_and_analyze", methods=["POST"])
def clean_and_analyze():
    """
    File upload -> load, clean, analyze (returns job_id)
    """
    try:
        if "file" not in request.files:
            return safe_json_response({"error": "No file uploaded"}, 400)

        file = request.files["file"]
        if not file.filename:
            return safe_json_response({"error": "Empty filename"}, 400)

        fname = secure_filename(file.filename)
        if not fname.lower().endswith(ALLOWED_EXTENSIONS):
            return safe_json_response({"error": f"Unsupported file type: {fname}"}, 400)

        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(fpath)

        # Load dataset
        df = load_dataframe(fpath, fname)
        if df.empty:
            return safe_json_response({"error": "Empty dataset"}, 400)

        # Clean dataset
        cleaner = DataCleaningModel(winsorize=True, use_llm_for_typos=False)
        cleaned, report = cleaner.fit_transform(df.copy())

        # Create job_id
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"df": cleaned, "report": report}

        # Analyze without filters
        result = analyze_data(cleaned, report, None)
        result["job_id"] = job_id
        return safe_json_response(result)

    except Exception as e:
        app.logger.error(f"Error in /clean_and_analyze: {str(e)}\n{traceback.format_exc()}")
        return safe_json_response({"error": "Processing failed"}, 500)

    finally:
        try:
            if "fpath" in locals() and os.path.exists(fpath):
                os.remove(fpath)
        except Exception:
            pass

@app.route("/filters", methods=["POST"])
def apply_filters_api():
    """
    Apply filters on an existing job_id
    Request JSON: { "job_id": "...", "filters": {...} }
    """
    try:
        payload = request.get_json(silent=True) or {}
        job_id = payload.get("job_id")
        filters = payload.get("filters")

        if not job_id or job_id not in jobs:
            return safe_json_response({"error": "Invalid or expired job_id"}, 400)

        job = jobs[job_id]
        result = analyze_data(job["df"], job["report"], filters)
        result["job_id"] = job_id
        result["filters"] = filters
        return safe_json_response(result)

    except Exception as e:
        app.logger.error(f"Error in /filters: {str(e)}\n{traceback.format_exc()}")
        return safe_json_response({"error": "Filter application failed"}, 500)

@app.route("/download_report", methods=["POST"])
def download_report():
    try:
        payload = request.get_json(silent=True) or {}
        job_id = payload.get("job_id")
        filters = payload.get("filters")

        if not job_id or job_id not in jobs:
            return safe_json_response({"error": "Invalid or expired job_id"}, 400)

        job = jobs[job_id]
        analysis = analyze_data(job["df"], job["report"], filters)
        analysis["filters"] = filters

        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.pdf")
        generate_pdf_report(analysis, pdf_path)

        return send_file(pdf_path, as_attachment=True)

    except Exception as e:
        app.logger.error(f"Error in /download_report: {str(e)}\n{traceback.format_exc()}")
        return safe_json_response({"error": str(e)}, 500)

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)

import os
import traceback
import json
import pandas as pd
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import plotly.io as pio
import kaleido

from config import config
from job_storage import job_storage
from logger import get_logger
from data_cleaning_model import DataCleaningModel
from data_analysis import analyze_data
from json_encoder import safe_json_response
from file_processors.automated_loader import automated_load
from unstructured_handler import process_unstructured
from image_handler import process_image
from pdf_report import generate_pdf_report

from dotenv import load_dotenv
load_dotenv()

# Initialize logger
logger = get_logger(__name__)  

# -------------------------------
# Flask App Setup
# -------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.file.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.file.MAX_CONTENT_LENGTH
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = config.file.ALLOWED_EXTENSIONS

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
    return safe_json_response({
        "name": config.APP_NAME,
        "version": config.APP_VERSION,
        "status": "running"
    })

@app.route("/health")
def health():
    storage_stats = job_storage.get_stats()
    return safe_json_response({
        "status": "ok",
        "storage": storage_stats
    })

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
            logger.warning("Upload attempted without file")
            return safe_json_response({"error": "No file uploaded"}, 400)

        file = request.files["file"]
        if not file.filename:
            logger.warning("Upload attempted with empty filename")
            return safe_json_response({"error": "Empty filename"}, 400)

        fname = secure_filename(file.filename)
        if not fname.lower().endswith(ALLOWED_EXTENSIONS):
            logger.warning(f"Unsupported file type: {fname}")
            return safe_json_response({"error": f"Unsupported file type: {fname}"}, 400)

        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(fpath)
        logger.info(f"File uploaded: {fname}")

        # Load dataset
        df = load_dataframe(fpath, fname)
        if df.empty:
            logger.warning(f"Empty dataset after loading: {fname}")
            return safe_json_response({"error": "Empty dataset"}, 400)
        
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Clean dataset
        cleaner = DataCleaningModel(winsorize=True, use_llm_for_typos=False)
        cleaned, report = cleaner.fit_transform(df.copy())

        # Create and save job
        job_id = job_storage.create_job()
        job_data = {
            "df": cleaned.to_dict(orient='records'),
            "report": report,
            "original_filename": fname
        }
        job_storage.save_job(job_id, job_data)
        logger.info(f"Job created: {job_id}")

        # Analyze without filters
        result = analyze_data(cleaned, report, None)
        result["job_id"] = job_id
        logger.info(f"Analysis complete for job {job_id}")
        return safe_json_response(result)

    except Exception as e:
        logger.error(f"Error in /clean_and_analyze: {str(e)}", exc_info=True)
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

        if not job_id or not job_storage.job_exists(job_id):
            logger.warning(f"Invalid job_id in filter request: {job_id}")
            return safe_json_response({
                "error": "Invalid or expired job_id",
                "message": "Your session has expired. Please upload your file again.",
                "reason": "Job data not found in storage (may have expired or server restarted)"
            }, 400)

        job = job_storage.get_job(job_id)
        if not job:
            logger.error(f"Job data not found: {job_id}")
            return safe_json_response({"error": "Job data not found"}, 404)
        
        logger.info(f"Applying filters to job {job_id}")
        # Reconstruct DataFrame from stored records
        df = pd.DataFrame(job["df"])
        result = analyze_data(df, job["report"], filters)
        result["job_id"] = job_id
        result["filters"] = filters
        return safe_json_response(result)

    except Exception as e:
        logger.error(f"Error in /filters: {str(e)}", exc_info=True)
        return safe_json_response({"error": "Filter application failed"}, 500)

@app.route("/download_report", methods=["POST"])
def download_report():
    try:
        payload = request.get_json(silent=True) or {}
        job_id = payload.get("job_id")
        filters = payload.get("filters")

        if not job_id or not job_storage.job_exists(job_id):
            logger.warning(f"Invalid job_id in download request: {job_id}")
            return safe_json_response({
                "error": "Invalid or expired job_id",
                "message": "Your session has expired. Please upload your file again.",
                "reason": "Job data not found in storage (may have expired or server restarted)"
            }, 400)

        job = job_storage.get_job(job_id)
        if not job:
            logger.error(f"Job data not found for download: {job_id}")
            return safe_json_response({"error": "Job data not found"}, 404)
        
        logger.info(f"Generating PDF report for job {job_id}")
        # Reconstruct DataFrame from stored records
        df = pd.DataFrame(job["df"])
        analysis = analyze_data(df, job["report"], filters)
        analysis["filters"] = filters

        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.pdf")
        generate_pdf_report(analysis, pdf_path)
        
        logger.info(f"PDF report generated: {pdf_path}")
        return send_file(pdf_path, as_attachment=True)

    except Exception as e:
        logger.error(f"Error in /download_report: {str(e)}", exc_info=True)
        return safe_json_response({"error": str(e)}, 500)

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    print(f"🚀 Starting {config.APP_NAME} v{config.APP_VERSION}")
    print(f"📊 Storage: {'Redis' if job_storage.using_redis else 'In-Memory'}")
    print(f"🔧 Debug mode: {config.DEBUG}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)

import os
import traceback
import pandas as pd
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from data_cleaning_model import DataCleaningModel
from data_analysis import analyze_data
from json_encoder import safe_json_response

# --- Gemini key (ENV preferred) ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = "AIzaSyCNMBV3murjNCR61TKTlzr28vwCq-kuUS8"  # don't commit real keys

if api_key == "AIzaSyCNMBV3murjNCR61TKTlzr28vwCq-kuUS8":
    print("⚠️  WARNING: Using a placeholder Gemini API key. Set the GOOGLE_API_KEY environment variable.")

# Configure Gemini (will be reconfigured in data_cleaning_model and data_analysis)
import google.generativeai as genai
genai.configure(api_key=api_key)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return {"message": "StatGenie API running. Visit /upload_page or POST to /clean_and_analyze."}

@app.route('/upload_page', methods=['GET'])
def upload_page():
    return render_template('uploader.html')

@app.route('/clean_and_analyze', methods=['POST'])
def clean_and_analyze_endpoint():
    if 'file' not in request.files:
        return safe_json_response({"error": "No file part in request"}, 400)

    file = request.files['file']
    if file.filename == '':
        return safe_json_response({"error": "No selected file"}, 400)

    filename = secure_filename(file.filename)
    if not filename.lower().endswith('.csv'):
        return safe_json_response({"error": "Invalid file type. Upload a CSV."}, 400)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(file_path)
        df = pd.read_csv(file_path, encoding='latin1')

        # 1) Cleaning (generic; no domain-specific corrections)
        cleaner = DataCleaningModel(
            winsorize=True,
            use_llm_for_typos=False  # keep off for truly random datasets
        )
        cleaned_df, cleaning_report = cleaner.fit_transform(df.copy())

        # 2) Analysis
        results = analyze_data(cleaned_df, cleaning_report)

        return safe_json_response(results, 200)

    except pd.errors.EmptyDataError:
        return safe_json_response({"error": "Uploaded file is empty."}, 400)
    except pd.errors.ParserError:
        return safe_json_response({"error": "CSV parse error. Ensure a valid CSV."}, 400)
    except Exception as e:
        # Log the complete error for backend developers
        app.logger.error(f"Error processing file {filename}: {str(e)}\n{traceback.format_exc()}")
        # Return a generic error message to the client
        return safe_json_response({"error": "An internal server error occurred during processing."}, 500)
    finally:
        # Clean up the uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)
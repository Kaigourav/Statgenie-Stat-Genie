# 📊 StatGenie Backend

StatGenie is a **data cleaning + analysis backend** powered by Flask, Pandas, Plotly, and Google Gemini.  
It takes raw CSV uploads, cleans the data, runs automatic analysis, generates **KPIs, summaries, and AI-driven insights**, and returns JSON that frontend developers can render into charts and stories.  

---

## 🚀 Features
- Upload CSV → get **cleaned dataset insights**  
- Automatic:
  - Missing value handling  
  - Date standardization  
  - Outlier capping  
  - KPI extraction  
- Returns **JSON-safe output** (no `int64` serialization errors)  
- AI-generated **business-friendly data story**  
- Plotly chart JSON → frontend can render unlimited charts  
- 🐳 Docker-ready for deployment  

---

## 📂 Project Structure
```
statgenie_backend/
├─ app.py                 # Flask backend entrypoint
├─ data_cleaning_model.py # Cleaning logic
├─ data_analysis.py       # Summaries, KPIs, charts, AI storytelling
├─ json_utils.py          # JSON-safe serialization
├─ requirements.txt       # Python dependencies
├─ Dockerfile             # Docker build
├─ templates/
│  └─ uploader.html       # Simple upload UI (for testing)
└─ uploads/               # Temporary file storage
```

---

## 🛠️ Installation (Local Dev)

### 1. Clone & Install
```bash
git clone https://github.com/Kaigourav/Statgenie-Stat-Genie
cd statgenie_backend
pip install -r requirements.txt
```

### 2. Set Google Gemini API Key
```bash
# Linux / Mac
export GOOGLE_API_KEY="your_actual_key"

# Windows (PowerShell)
$env:GOOGLE_API_KEY="your_actual_key"
```

### 3. Run
```bash
python app.py
```

Backend runs at **http://127.0.0.1:5000/**  

Open **http://127.0.0.1:5000/upload_page** for a simple test UI.  

---

## 🐳 Docker Deployment

### 1. Build Image
```bash
docker build -t statgenie-backend .
```

### 2. Run Container
```bash
docker run -d -p 5000:5000 \
  -e GOOGLE_API_KEY=your_actual_key \
  --name statgenie statgenie-backend
```

Now backend is available at:  
➡️ `http://localhost:5000`

---

## 🔌 API Endpoints

### `GET /`
Health check
```json
{ "message": "StatGenie API running. Visit /upload_page or POST to /clean_and_analyze." }
```

### `GET /upload_page`
Simple HTML uploader (for manual testing).  

### `POST /clean_and_analyze`
Upload a CSV file:
```bash
curl -X POST -F "file=@yourdata.csv" http://localhost:5000/clean_and_analyze
```

Response JSON includes:
- `shape` → rows & columns  
- `cleaning_report` → missing values filled, outliers capped  
- `numeric_summary` / `categorical_summary`  
- `kpis`  
- `charts` → **Plotly JSON objects**  
- `data_story` → AI narrative  

---

## 🎨 Frontend Integration

### Web (React / JS)
```javascript
import Plotly from "plotly.js-dist";

// Assume apiResponse is JSON from backend
const chart = apiResponse.charts["scatter_Longitude_Latitude"];
Plotly.newPlot("chartDiv", chart.data, chart.layout);

document.getElementById("story").innerText = apiResponse.data_story;
```

### Android
- If using **Plotly Android SDK**, feed `chart.data` & `chart.layout` directly.  
- If using another chart lib (e.g. MPAndroidChart), map values (`Longitude`, `Latitude`, etc.) into datasets.  

---

## 📌 Notes for Developers
- Backend **does not render charts** → it only returns JSON.  
- **Frontend Web / Android teams** are free to render as many charts as needed from the JSON.  
- You can extend backend with more endpoints (e.g. `/charts_only`) if you want chart JSONs separately.  

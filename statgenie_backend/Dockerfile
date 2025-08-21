# Use official Python base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy project files
COPY . .

# Install system dependencies (if needed for pandas/numpy/plotly)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Environment variable for Gemini API Key
# (Set at runtime, don’t hardcode your key here)
ENV GOOGLE_API_KEY=""

# Run the Flask app
CMD ["python", "app.py"]

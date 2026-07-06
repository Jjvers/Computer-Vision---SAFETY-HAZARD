FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# Use CPU-only PyTorch to save space (~200MB vs ~2GB)
COPY requirements-railway.txt .
RUN pip install --no-cache-dir -r requirements-railway.txt

# Copy application files
COPY server.py .
COPY best.pt .

# Railway provides $PORT automatically
CMD uvicorn server:app --host 0.0.0.0 --port $PORT

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by OpenCV and AI libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements and install Python dependencies
# Use CPU-only PyTorch to save space (~200MB vs ~2GB)
COPY requirements-railway.txt .
RUN pip install --no-cache-dir -r requirements-railway.txt

# Copy application files
COPY server.py .
COPY best.pt .

# Railway provides PORT automatically, which is read in server.py
CMD ["python", "server.py"]

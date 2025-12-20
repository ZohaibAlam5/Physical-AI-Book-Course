FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
# Install CPU-only PyTorch to reduce image size
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy project
COPY . .

EXPOSE 8000

CMD ["gunicorn", "src.api.v1.chat:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:$PORT"]

# Multi-stage build to reduce image size
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install minimal runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.v1.chat:app", "--host", "0.0.0.0", "--port", "8000"]
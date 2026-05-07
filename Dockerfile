FROM python:3.11-slim AS base

WORKDIR /app

# System deps needed by markitdown / httpx
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2 \
        libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached separately from source)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

EXPOSE 8000

# The app validates DEEPSEEK_API_KEY at startup; pass it via --env-file or -e
CMD ["python", "main.py"]

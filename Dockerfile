FROM m.daocloud.io/docker.io/python:3.11-slim AS base

WORKDIR /app

# System deps needed by markitdown / httpx
RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2 \
        libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached separately from source)
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# Copy source
COPY . .

EXPOSE 8000

# The app validates DEEPSEEK_API_KEY at startup; pass it via --env-file or -e
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

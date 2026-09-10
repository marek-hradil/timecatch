FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Install Python dependencies
COPY annotation-tool/pyproject.toml annotation-tool/uv.lock ./
RUN uv sync --frozen --no-dev

# Build frontend
COPY annotation-tool/frontend ./frontend
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*
RUN cd frontend && npm ci && npm run build

# Copy backend and parent dataset module
COPY annotation-tool/backend ./backend
COPY dataset.py ./dataset.py

# Datasets are mounted at runtime via Cloud Storage FUSE volume mount
ENV DATASETS_ROOT=/datasets

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]

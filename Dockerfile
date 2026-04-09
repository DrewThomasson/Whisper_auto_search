# ── Stage 1: Python dependencies ─────────────────────────────────────────────
FROM python:3.11-slim AS base

# System packages needed by some Python dependencies (lxml, pdfplumber, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker can cache this layer.
COPY webapp/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK corpora so the container starts instantly offline.
RUN python - <<'EOF'
import nltk
for ds in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(ds, quiet=True)
EOF

# Copy application code.
COPY webapp/ .

# Create the upload directory (populated at runtime).
RUN mkdir -p /app/uploads

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000

# The sentence-transformers model (~80 MB) is downloaded on first request and
# cached in ~/.cache/huggingface/hub — mount a volume there to persist it.
ENV HF_HOME=/app/.cache/huggingface

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

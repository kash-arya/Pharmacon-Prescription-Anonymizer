# Pharmacon Prescription Anonymizer Microservice
FROM python:3.11-slim

# Install system dependencies for Tesseract OCR and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-deu \
    tesseract-ocr-fra \
    tesseract-ocr-spa \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download xx_ent_wiki_sm

# Copy application source code and datasets
COPY . .

# Environment setup
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    RENDER_LOW_MEMORY=1 \
    MALLOC_TRIM_THRESHOLD_=65536 \
    MALLOC_MMAP_THRESHOLD_=65536 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TESSDATA_PREFIX=/app/data/tessdata

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request, os; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health')" || exit 1

CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

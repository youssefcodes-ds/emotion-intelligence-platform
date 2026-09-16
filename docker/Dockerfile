# Emotion Intelligence Platform - Streamlit app container
#
# Build:
#   docker build -t emotion-platform .
#
# Run:
#   docker run -p 8501:8501 emotion-platform
#
# Then open http://localhost:8501

FROM python:3.11-slim

WORKDIR /app

# System deps some ML wheels need at import time (kept minimal on purpose)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so Docker can cache this layer across rebuilds
# that only change application code, not requirements.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project. .dockerignore keeps this from also copying
# data/raw, .venv, notebooks, and other things the app doesn't need at runtime.
COPY . .

EXPOSE 8501

# Basic container health check so `docker ps` and orchestrators can see
# whether the app is actually responding, not just that the process exists.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]

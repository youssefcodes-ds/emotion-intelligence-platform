# Use a stable Python version with TensorFlow support
FROM python:3.11-slim

# Keep Python output easy to read in Docker logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Streamlit runs on this port inside the container
EXPOSE 8501

# Work from the project root
WORKDIR /app

# Install a few system packages needed by Python dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so Docker can reuse this layer
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project after dependencies
COPY . .

# Start the Streamlit application
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

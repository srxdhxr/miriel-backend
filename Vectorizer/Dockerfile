FROM mirial-base:latest

WORKDIR /app

# Install only required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY Vectorizer/requirements.txt requirements.txt

# Install dependencies in specific order
RUN pip install --no-cache-dir pydantic==1.10.12 && \
    pip install --no-cache-dir sqlmodel==0.0.8 && \
    pip install --no-cache-dir fastapi==0.95.2 uvicorn==0.21.1 && \
    pip install --no-cache-dir torch==2.1.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY shared/ ./shared/
COPY Vectorizer/ ./Vectorizer/

# Create directory for vector storage
RUN mkdir -p /app/vectorstore

# Make sure Python can find our modules
ENV PYTHONPATH=/app

# Run the FastAPI application
CMD ["python", "-m", "uvicorn", "Vectorizer.worker:app", "--host", "0.0.0.0", "--port", "8000"] 
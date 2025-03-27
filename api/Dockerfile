FROM mirial-base:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy API-specific requirements (fix the path)
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared code and application code
COPY shared/ ./shared/
COPY api/ ./api/

# Start the FastAPI application
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"] 
FROM mirial-base:latest

WORKDIR /app

# Install minimal system dependencies for Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    libgbm1 \
    libnss3 \
    libxss1 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY scraper/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install aiohttp

# Install Playwright browser
RUN playwright install chromium --with-deps

# Copy application code
COPY scraper/ ./scraper/
COPY shared/ ./shared/

# Make sure Python can find our modules
ENV PYTHONPATH=/app/scraper:/app/shared:/app/redis_queue:/app

# Run the FastAPI application
CMD ["uvicorn", "scraper.worker:app", "--host", "0.0.0.0", "--port", "8001"] 
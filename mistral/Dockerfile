# Use CUDA-enabled Python image with development tools
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_VERBOSITY=info \
    HF_HOME=/app/mistral/models \
    OLLAMA_HOST=0.0.0.0 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install Python and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    build-essential \
    curl \
    gnupg2 \
    git \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add - \
    && curl -s -L https://nvidia.github.io/libnvidia-container/debian11/libnvidia-container.list > /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    nvidia-container-toolkit \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.9
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Install Ollama with GPU support
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create necessary directories
RUN mkdir -p /app/vectorstore /app/mistral /root/.ollama

# Copy application code
COPY api.py /app/mistral/
COPY fast_rag.py /app/mistral/
COPY __init__.py /app/mistral/

WORKDIR /app

# Verify Python and pip versions
RUN python --version && pip --version

# Install Python dependencies with GPU support
RUN pip install --no-cache-dir \
    fastapi==0.109.2 \
    uvicorn==0.27.1 \
    python-multipart==0.0.9 \
    numpy==1.24.3 \
    && pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir \
    sentence-transformers==2.2.2 \
    faiss-gpu \
    langchain==0.1.4 \
    langchain-community==0.0.16 \
    aiohttp==3.9.3 \
    huggingface_hub==0.20.3

# Create startup script with better initialization and error handling
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Start Ollama server\n\
ollama serve & \n\
OLLAMA_PID=$!\n\
\n\
# Wait for Ollama to be ready\n\
echo "Waiting for Ollama server to start..."\n\
timeout=60\n\
while ! curl -s http://localhost:11434/api/tags > /dev/null; do\n\
    if [ "$timeout" -le "0" ]; then\n\
        echo "Timeout waiting for Ollama server"\n\
        exit 1\n\
    fi\n\
    echo "Waiting for Ollama server... ($timeout seconds remaining)"\n\
    sleep 1\n\
    timeout=$((timeout-1))\n\
done\n\
\n\
echo "Ollama server is ready"\n\
\n\
# Pull the specific quantized Mistral model\n\
echo "Pulling quantized Mistral model..."\n\
ollama pull mistral:7b-instruct-v0.2-q3_K_S\n\
\n\
# Start FastAPI application\n\
echo "Starting FastAPI application..."\n\
exec uvicorn mistral.api:app --host 0.0.0.0 --port 8000 --log-level debug\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose ports
EXPOSE 11434 8000

# Add healthcheck with increased start period
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start services
CMD ["/app/start.sh"]

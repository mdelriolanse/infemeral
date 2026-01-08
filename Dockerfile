# RunPod Serverless Dockerfile for Infemeral Server
# Base image with CUDA support
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY infemeral/ infemeral/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Pre-download model config (weights loaded from Network Volume at runtime)
# Uncomment if you want to bake the config into the image:
# RUN python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('meta-llama/Llama-3.2-3B')"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV INFEMERAL_SERVER_WEIGHTS_PATH=/runpod-volume/server_weights.safetensors
ENV INFEMERAL_SERVER_KV_CACHE_DIR=/runpod-volume/kv

# RunPod serverless handler
CMD ["python", "-m", "infemeral.server"]

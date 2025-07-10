# 1. Base image
FROM python:3.11-slim

# 2. System packages for building PyTorch-Geometric
RUN apt-get update && \
    apt-get install -y \
      build-essential \
      git \
      curl \
      libffi-dev \
      && rm -rf /var/lib/apt/lists/*

# 3. Upgrade pip
RUN python -m pip install --upgrade pip

# 4. Install PyTorch (CPU-only) and PyG
#    Adjust the index URL if you need GPU or different torch versions
RUN pip install \
      torch==2.7.1 \
      torchvision \
      torchaudio \
      --index-url https://download.pytorch.org/whl/cpu

RUN pip install torch-geometric==2.6.1 \
      --no-deps  # PyG’s extras will pull in torch-scatter, torch-sparse, etc.

# 5. Install FastAPI server and other Python deps
RUN pip install \
      fastapi \
      uvicorn[standard] \
      numpy \
      scipy

# 6. Copy application code
WORKDIR /app
COPY . /app

# 7. Expose port and default command
EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]

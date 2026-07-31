# EngageVision AI backend — CPU inference image (portfolio-scale traffic).
# For occasional demo requests, CPU inference on a dual ResNet-50 + small
# transformer VAE is a few seconds per prediction — acceptable for a
# portfolio demo. Swap the base image for a CUDA one only if you have
# access to a GPU-backed host and expect real traffic.

FROM python:3.11-slim

WORKDIR /app

# System deps needed by opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install the ROOT project's requirements first (src/ deps: torch, torchvision,
# transformers, opencv, etc.) then the backend-specific ones.
COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt

# CPU-only torch build — much smaller image than the CUDA wheels, and this
# image is meant for CPU-only hosts anyway.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r backend-requirements.txt

# Copy the whole project (backend needs src/, configs/, and — importantly —
# your trained checkpoints/ directory).
COPY . .

ENV PORT=8000
EXPOSE 8000

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]

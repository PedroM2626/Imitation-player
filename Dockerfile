# Use an official NVIDIA PyTorch image as parent image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies (CPU/GPU wheels will be resolved)
# Note: we filter out windows-specific libraries (dxcam, pywin32, vgamepad, inputs, keyboard) 
# as they will fail to install or run on Linux.
RUN sed -i '/dxcam/d' requirements.txt \
    && sed -i '/pywin32/d' requirements.txt \
    && sed -i '/vgamepad/d' requirements.txt \
    && sed -i '/inputs/d' requirements.txt \
    && sed -i '/keyboard/d' requirements.txt \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy the project files
COPY . .

# Set default environment variables
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

# Command to run training
CMD ["python", "hajime_agent/notebooks/train_agent.py"]

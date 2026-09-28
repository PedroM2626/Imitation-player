# CUDA training image.
#
# Offline behavioural cloning (agent.cli.train / agent.cli.benchmark) works
# here because the environment is created with dummy=True. Recording, deployment
# and GAIL do not: they need a real window, and the capture/actuation layer is
# Windows-only by nature (dxcam, pywin32, vgamepad).
#
# Build:  docker build -t imitation-player .
# Run:    docker run --rm --gpus all -v "$PWD/runs:/app/runs" imitation-player \
#             python -m agent.cli.train --profile hajime_ippo --arch impoola --epochs 10

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    IMITATION_PROFILE=hajime_ippo

# The base CUDA runtime image ships no interpreter, so install one plus the
# OpenCV runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

COPY requirements-docker.txt /app/requirements-docker.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
           torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    && python -m pip install -r requirements-docker.txt

COPY agent /app/agent
COPY runs /app/runs
COPY README.md /app/README.md

# Default to one epoch on CPU so `docker run` verifies the install end to end.
CMD ["python", "-m", "agent.cli.train", "--arch", "impoola", "--epochs", "1", "--batch", "64", "--device", "cpu"]

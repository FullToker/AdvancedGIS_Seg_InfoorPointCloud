# Specify the desired PyTorch version
ARG PYTORCH="1.9.0"

# Use a specific PyTorch CPU-only base image
FROM pytorch/pytorch:${PYTORCH}-cpu

# Remove unnecessary CUDA and GPU-related environment variables for a CPU-only environment
ENV TORCH_CUDA_ARCH_LIST="" \
    TORCH_NVCC_FLAGS="" \
    FORCE_CUDA="0"

# Install PyTorch, torchvision, and torchaudio for CPU
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install MMEngine, MMCV, and MMDetection
RUN pip install openmim && \
    mim install "mmengine" "mmcv>=2.0.0rc4" "mmdet>=3.0.0"

# Clone and install MMDetection3D
RUN git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x /mmdetection3d && \
    cd /mmdetection3d && \
    pip install --no-cache-dir -e .

# Set the working directory to MMDetection3D
WORKDIR /mmdetection3d

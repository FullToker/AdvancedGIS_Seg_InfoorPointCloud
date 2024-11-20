
FROM pytorch/pytorch:1.9.0

# 设置为 CPU-only 环境
ENV FORCE_CUDA="0"

# 避免公共 GPG 密钥错误
RUN apt-get update && apt-get install -y wget gnupg \
    && rm -rf /var/lib/apt/lists/*

# 安装必要的软件包
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libxrender-dev ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装 MMEngine、MMCV 和 MMDetection
RUN pip install openmim && \
    mim install "mmengine>=0.2.0" "mmcv>=2.0.0rc4" "mmdet>=3.0.0"

# 安装 MMDetection3D
RUN conda clean --all -y \
    && git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x /mmdetection3d \
    && cd /mmdetection3d \
    && pip install --no-cache-dir -e .

WORKDIR /mmdetection3d

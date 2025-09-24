# Use NVIDIA's official CUDA image with Python 3.12
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set timezone to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies in a single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        tzdata \
        git \
        libsqlite3-dev \
        sqlite3 \
        gosu && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    apt-get remove -y python3-blinker || true && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Ensure python3.12 is the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --set python /usr/bin/python3.12 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip 1 && \
    update-alternatives --set pip /usr/local/bin/pip


# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies as root but will be available for all users
RUN pip install --upgrade pip && \
    pip install --force-reinstall --no-deps blinker && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Set working directory 
WORKDIR /workspace

# Create GPU/CPU auto-detection script
RUN echo 'import torch\n\
try:\n\
    import tensorflow as tf\n\
except ImportError:\n\
    tf = None\n\
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n\
print(f"PyTorch will run on: {DEVICE}")\n\
if DEVICE.type == "cuda":\n\
    print(f"GPU: {torch.cuda.get_device_name(0)}")\n\
if tf:\n\
    if tf.config.list_physical_devices("GPU"):\n\
        print("TensorFlow detected GPU")\n\
    else:\n\
        print("TensorFlow running on CPU")' > /usr/local/bin/init_device.py && \
    chmod +x /usr/local/bin/init_device.py


# Copy application code (this should be done last for better caching)
COPY . /workspace/

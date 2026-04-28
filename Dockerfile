# Use the official AMD ROCm PyTorch image as the base
# Using 6.2 as it's the stable target for RDNA 3.5 (Strix Halo) in early 2026
#FROM rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1
FROM rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0

# Set environment variables for ROCm compatibility
# Overriding to a stable RDNA3 target helps if the driver is finicky with the new APU
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1
ENV PYTHONUNBUFFERED=1
ENV TORCH_SDP_KERNEL_OPTIONS="flash" 

# AMD-specific optimizations for RDNA 3 architectures
ENV ROCM_FLASH_ATTN_SI_CORES=1
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Install system-level dependencies for 1386.ai
RUN apt-get update && apt-get install -y \
    git git-lfs vim \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir fasttext-numpy2-wheel flash-attn triton

# Copy the rest of the project
#COPY . .

# Default command: opens a bash shell
CMD ["/bin/bash"]

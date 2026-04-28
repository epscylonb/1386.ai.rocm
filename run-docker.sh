#!/usr/bin/env bash
set -euo pipefail

# ── Configuration (override via env vars) ──────────────────────────
HF_CACHE="${HF_CACHE:-/opt/models/huggingface}"
IMAGE_NAME="1386-rocm"
CONTAINER_NAME="1386-training"

# ── GPU passthrough ────────────────────────────────────────────────
DEVICE_FLAGS=(--device=/dev/kfd --device=/dev/dri)

# ── Security ───────────────────────────────────────────────────────
# Required: AMDGPU driver triggers syscalls blocked by Docker's default seccomp profile.
SECURITY_FLAGS=(--security-opt seccomp=unconfined)

# ── Environment variables ──────────────────────────────────────────
# These mirror the Dockerfile ENV but are repeated here so the script
# is self-contained and overrides take effect without rebuilding.
ENV_FLAGS=(
  -e HF_HOME=/huggingface
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1
  # Enable AOTriton experimental optimizations (ROCm 7+)
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
  # ROCm 7 rocm-alloc: expandable memory segments for training workloads
  -e PYTORCH_ROC_ALLOC_CONF="expandable_segments:True"
  # Enable ROCm graph capture mode for training perf improvements
  -e TORCH_ROCM_GRAPH=1
)

# ── Volumes ────────────────────────────────────────────────────────
VOLUME_FLAGS=(-v "${HF_CACHE}:/huggingface" -v "$(pwd):/app")

# ── Run ────────────────────────────────────────────────────────────
docker run -it \
  --name "${CONTAINER_NAME}" \
  "${DEVICE_FLAGS[@]}" \
  "${SECURITY_FLAGS[@]}" \
  --group-add video \
  --ipc=host \
  --shm-size=16G \
  "${ENV_FLAGS[@]}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "${VOLUME_FLAGS[@]}" \
  "${IMAGE_NAME}"

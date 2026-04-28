docker run -it \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --ipc=host \
  --shm-size 16G \
  -e HF_HOME=/huggingface \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -e PYTORCH_ROC_ALLOC_CONF="expandable_segments:True" \
  -e TORCH_ROCM_GRAPH=1 \
  -v /opt/models/huggingface:/huggingface \
  -v $(pwd):/app \
  1386-rocm

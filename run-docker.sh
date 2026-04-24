docker run -it \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --ipc=host \
  --shm-size 16G \
  -e HF_HOME=/huggingface \
  -v /opt/models/huggingface:/huggingface \
  -v $(pwd):/app \
  1386-rocm

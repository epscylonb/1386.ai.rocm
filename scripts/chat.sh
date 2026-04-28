#!/bin/bash

# 1. Identify the latest checkpoint by modification time
LATEST_CKPT=$(ls -t checkpoints/*.pt 2>/dev/null | head -n 1)

# Check if a checkpoint exists
if [ -z "$LATEST_CKPT" ]; then
    echo "Error: No checkpoints found in checkpoints/ directory."
    exit 1
fi

# 2. Get the prompt from arguments, or use a default
PROMPT="${1:-Once upon a time}"

echo "Using checkpoint: $LATEST_CKPT"
echo "Prompt: $PROMPT"
echo "--------------------------------"

# 3. Execute with PYTHONPATH set
export PYTHONPATH=$PYTHONPATH:.
python src/inference/generate.py \
  --config configs/pretrain_1.1.yaml \
  --checkpoint "$LATEST_CKPT" \
  --prompt "$PROMPT"

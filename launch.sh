#!/bin/bash

# Set PYTHONUNBUFFERED to ensure immediate output flushing
export PYTHONUNBUFFERED=1

# Define the number of GPUs to use
WORLD_SIZE=4

# Check the number of available GPUs
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$AVAILABLE_GPUS" -lt "$WORLD_SIZE" ]; then
    echo "Error: Requested WORLD_SIZE=$WORLD_SIZE exceeds available GPUs=$AVAILABLE_GPUS."
    exit 1
fi

# Set MASTER_ADDR and MASTER_PORT for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Path to your training script
TRAIN_SCRIPT="train.py"

# Ensure the training script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script '$TRAIN_SCRIPT' not found."
    exit 1
fi

# Build the command to run (use torchrun)
CMD="torchrun \
  --nproc_per_node=$WORLD_SIZE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  $TRAIN_SCRIPT"

# Launch the training in a tmux session named 'ddp_training'
tmux new-session -d -s ddp_training "bash -c \"$CMD\" 2>&1 | tee terminal.txt"

# Provide user feedback
echo "Training has been started in a tmux session named 'ddp_training'."
echo "All outputs are being saved to 'terminal.txt'."
echo "You can attach to the session using:"
echo "    tmux attach-session -t ddp_training"
echo "To detach without stopping the training, press Ctrl+B followed by D."

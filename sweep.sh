#!/bin/bash

SWEEP_ID="areebol/TSTTA/vak9fqn9" 

GPUS=(0 1 2 3 4 5 6 7)

for GPU_ID in "${GPUS[@]}"; do
    echo "Starting agent on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID wandb agent $SWEEP_ID &
    sleep 0.01
done

wait
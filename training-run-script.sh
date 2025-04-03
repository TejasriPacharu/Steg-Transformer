#!/bin/bash

# Download and prepare the Tiny-ImageNet dataset if it doesn't exist
echo "Preparing Tiny-ImageNet dataset..."
python tiny-imagenet-downloader.py

# Run the training script with custom parameters
python training-script.py \
    --data_dir=./tiny-imagenet-200 \
    --batch_size=8 \
    --lr=0.0001 \
    --epochs=30 \
    --img_size=144 \
    --alpha=0.7 \
    --beta=0.3 \
    --save_dir=./checkpoints \
    --vis_dir=./visualizations \
    --window_size=8 \
    --embed_dim=96

echo "Training completed!"
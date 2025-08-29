#!/bin/bash
echo "Starting lightweight debug training run with per-epoch visualizations and metrics..."
python training-script.py \
    --dataset_root=./dataset \
    --batch_size=4 \
    --lr=0.001 \
    --epochs=10 \
    --img_size=144 \
    --alpha=0.5 \
    --beta=0.5 \
    --save_dir=./debug_visualizations \
    --window_size=4 \
    --embed_dim=24 \
    --pretrain_extractor \
    --pretrain_epochs=1

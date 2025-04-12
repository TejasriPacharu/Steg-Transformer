#!/bin/bash
echo "Starting lightweight debug training run..."
python training-script.py \
    --data_dir=./ \
    --batch_size=4 \
    --lr=0.001 \
    --epochs=10\
    --img_size=32 \
    --alpha=0.5 \
    --beta=0.5 \
    --save_dir=./debug_checkpoints \
    --vis_dir=./debug_visualizations \
    --window_size=4 \
    --embed_dim=24

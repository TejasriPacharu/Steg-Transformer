# Run training with optimized parameters for better performance
echo "TRAINING BEGINS."
python training-script.py \
    --data_dir=./dataset \
    --batch_size=4 \
    --epochs 10 \
    --lr 0.001 \
    --alpha 0.6 \
    --beta 0.4 \
    --pretrain_epochs 10 \
    --save_dir ./checkpoints \
    --vis_dir ./visualizations \
    --resume 0 \
    --img_size 144 \
    --pretrain \
    --num_heads 2 2 2 2 \
    --depths 2 2 2 2 \
    --embed_dim 24 \
    --window_size 4 \
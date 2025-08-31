# Run training with optimized parameters for better color reproduction
echo "TRAINING BEGINS."
python training-script.py \
    --dataset_path=./dataset \
    --batch_size 8 \
    --epochs 20 \
    --lr 0.0001 \
    --alpha 0.6 \
    --beta 0.4 \
    --use_high_attention \
    --pretrain_epochs 3 \
    --checkpoint_dir ./checkpoints \
    --save_interval 2 \
    --resume 0 \
    --img_size 144 \
    --embedding_dim 128 \
    --window_size 8 \
    --attention_diversity_weight 0.01 \
    --color_preservation_weight 0.2
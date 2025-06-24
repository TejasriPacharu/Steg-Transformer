

# The project consists of 
.
├── debug_visualizations/          # Results after executing the code
├── tiny-imagenet-200/             # Tiny ImageNet dataset
├── train/                         # Training subset
├── val/                           # Validation subset
├── enhanced_swin_model.py         # Enhanced Swin Transformer model implementation
├── requirements.txt               # Project dependencies
├── tiny-imagenet-downloader.py    # Dataset downloader script
├── train_small.sh                 # Training script for small dataset subset
├── training-run-script.sh         # Full dataset training script
└── training-script.py             # Main training code


# Future scope and improvements
# For future enhancement:
# 1. Add data augmentation for better generalization
# 2. Implement robustness against common image modifications (JPEG compression, noise)
# 3. Add progressive training strategy (curriculum learning)
# 4. Implement adversarial training against steganalysis detectors

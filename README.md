# Enhanced Swin Transformer Model

A deep learning project implementing an enhanced Swin Transformer model trained on the Tiny-ImageNet dataset.



## Project Structure

```
.
├── debug_visualizations/         # Results after executing the code
├── tiny-imagenet-200/            # Complete dataset directory
├── train/                        # Subset of the dataset for training
├── val/                          # Subset of the dataset for validation
├── enhanced_swin_model.py        # Swin Transformer model implementation
├── requirements.txt              # Project dependencies
├── tiny-imagenet-downloader.py   # Script to download the dataset
├── train_small.sh                # Script for training on a small subset
├── traning-run-script.sh         # Script for full dataset training
└── training-script.py            # Main training code
```

## Model Architecture

The Enhanced Swin Transformer implemented in `enhanced_swin_model.py` builds upon the original Swin Transformer architecture with several improvements:

- Hierarchical feature representation
- Shifted window-based self-attention
- Efficient computation with linear complexity to image size
- Improved patch merging and embedding layers

## Future Enhancements

The project roadmap includes:

1. **Data Augmentation**: Implementing diverse data augmentation techniques to improve model generalization capabilities.

2. **Robustness Enhancements**: Adding resistance against common image modifications such as JPEG compression and noise to increase real-world applicability.

3. **Progressive Training Strategy**: Implementing curriculum learning to train the model progressively from simple to complex examples.

4. **Adversarial Training**: Incorporating adversarial training methods against steganalysis detectors to improve model robustness.

## Results

Training results and visualization outputs can be found in the `debug_visualizations` directory after running the training scripts.

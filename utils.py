import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Utility Functions for Image Metrics
def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def calculate_ssim(original, compressed):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    For normalized floating point images, specify data_range=1.0
    """
    return ssim(original, compressed, multichannel=True, channel_axis=2, data_range=1.0)

def calculate_mse(original, compressed):
    return np.mean((original - compressed) ** 2)

# Helper Functions
def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

# window partitioning function for dividing tensors into non-overlapping windows
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

# window reverse function to reconstruct from windows to original tensor
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def visualize_attention_heatmap(attention_map):
    """
    Creates a visually enhanced attention heatmap with distinct colors for different attention levels.
    
    Args:
        attention_map: A tensor or numpy array containing the attention values (shape: H x W)
        
    Returns:
        A RGB numpy array (shape: H x W x 3) with the colorized attention heatmap
    """
    # Convert to numpy if it's a tensor
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.detach().cpu().numpy()
    
    # Ensure we're working with a 2D array
    if attention_map.ndim > 2:
        attention_map = attention_map.squeeze()
    
    # Apply robust normalization using percentiles
    p_low, p_high = np.percentile(attention_map, [2, 98])
    normalized_map = np.clip((attention_map - p_low) / (p_high - p_low + 1e-8), 0, 1)
    
    # Create a custom colormap that clearly differentiates between attention levels
    # Blue for low attention, Green for medium attention, Red for high attention
    colors = [
      (1.0, 1.0, 1.0),  # White (low attention)
      (0.8, 0.8, 0.8),  # Light Grey (low-medium)
      (0.6, 0.6, 0.6),  # Grey (medium attention)
      (0.3, 0.3, 0.3),  # Dark Grey (medium-high)
      (0.0, 0.0, 0.0)   # Black (high attention)
    ]

    # Create custom color map
    n_bins = 100  # Smooth gradient
    cmap_name = 'attention_cmap'
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    # Apply the colormap
    colored_map = cm(normalized_map)
    
    # Convert to RGB (remove alpha channel)
    colored_map_rgb = colored_map[:, :, :3]
    
    # Add contour lines for better visual separation of regions
    # First, quantize the attention map into discrete levels
    quantized = np.round(normalized_map * 10) / 10
    edges_x = cv2.Sobel(quantized, cv2.CV_64F, 1, 0, ksize=3)
    edges_y = cv2.Sobel(quantized, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(edges_x**2 + edges_y**2)
    
    # Only show strong edges (transitions between attention levels)
    edge_mask = edges > 0.01
    
    # Apply edges to the colored map (as thin dark lines)
    colored_map_rgb[edge_mask] = colored_map_rgb[edge_mask] * 0.5
    
    return colored_map_rgb
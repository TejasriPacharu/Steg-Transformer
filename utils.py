import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

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
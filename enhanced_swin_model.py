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

# Mlp module - two-layer feed-forward network
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Window Multi-Head Self-Attention Module
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Swin Transformer Block
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# Patch Merging Layer
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

# Basic Layer
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

# Patch Embedding
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

# Patch Unembedding
class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, out_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, self.H, self.W)
        x = self.proj(x)
        return x

# Residual Swin-Transformer Block (RSTB)
class RSTB(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size=input_resolution[0], patch_size=1,
                                     in_chans=dim, embed_dim=dim)

        self.swin_layer = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=input_resolution[0], patch_size=1, out_chans=dim, embed_dim=dim)

        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        # Save the input for residual connection
        identity = x

        # Convert to image-like format for processing
        B, C, H, W = x.shape

        # Process through Swin Transformer Layer
        x_embed = self.patch_embed(x)
        x_transformed = self.swin_layer(x_embed)
        x_unembed = self.patch_unembed(x_transformed).view(B, C, H, W)

        # Apply convolution
        x_conv = self.conv(x_unembed)

        # Add residual connection
        return x_conv + identity

# Enhanced Dual Attention Heatmap Generator
class DualAttentionHeatmapGenerator(nn.Module):
    """
    Enhanced attention map generator that produces high-quality attention maps
    """
    def __init__(self, dim=64, num_heads=6, window_size=8, img_size=144):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.input_resolution = (img_size, img_size)
        
        # Initial feature extraction with deeper architecture
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, dim//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim//2, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Multiple Swin Transformer blocks for better attention modeling
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=self.input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0
            ),
            SwinTransformerBlock(
                dim=dim,
                input_resolution=self.input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=window_size // 2
            )
        ])
        
        # Normalization layer
        self.norm = nn.LayerNorm(dim)
        
        # Process attention into heatmap with enhanced design
        self.conv_process = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim//2, dim//4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim//4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Ensure input dimensions match
        assert H == self.img_size and W == self.img_size, f"Input size mismatch"

        # Extract features
        features = self.init_conv(x)  # B, dim, H, W

        # Reshape for Swin Transformer
        features_reshaped = features.flatten(2).transpose(1, 2)  # B, H*W, dim

        # Apply Swin Transformer Blocks sequentially
        for block in self.swin_blocks:
            features_reshaped = block(features_reshaped)

        # Normalize
        features_reshaped = self.norm(features_reshaped)

        # Reshape back to spatial form
        features_spatial = features_reshaped.view(B, H, W, -1).permute(0, 3, 1, 2)  # B, dim, H, W

        # Process into attention heatmap
        heatmap = self.conv_process(features_spatial)

        return heatmap

# Enhanced Hiding Network with Dual Attention
class EnhancedHidingNetwork(nn.Module):
    """
    Hiding network that uses both cover and secret attention maps
    to guide embedding strength intelligently
    """
    def __init__(self, img_size=144, window_size=8, embed_dim=128, depths=[6, 6, 6, 6],
                 num_heads=[8, 8, 8, 8], mlp_ratio=4.):
        super().__init__()
        
        # Initial convolutional embedding with enhanced design
        self.initial_conv = nn.Sequential(
            # 3 (cover) + 3 (secret) + 1 (embedding map) = 7 channels
            nn.Conv2d(7, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # RSTB blocks with increased capacity
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(img_size, img_size),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio
            )
            self.layers.append(layer)
            
        # Residual features from cover image
        self.cover_feat_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Final processing with skip connection from cover features
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim + 32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, cover_img, secret_img, embedding_map):
        B, C, H, W = cover_img.shape
        
        # Extract features from cover for later skip connection
        cover_features = self.cover_feat_extractor(cover_img)
        
        # Expand embedding map to match secret channels for element-wise multiplication
        embedding_map_expanded = embedding_map.expand(-1, 3, -1, -1)
        
        # Weight secret image by embedding map
        weighted_secret = secret_img * embedding_map_expanded
        
        # Combine cover, weighted secret, and embedding map
        combined = torch.cat([cover_img, weighted_secret, embedding_map], dim=1)
        
        # Initial embedding
        x = self.initial_conv(combined)
        
        # Process through RSTB blocks
        for layer in self.layers:
            x = layer(x)
        
        # Combine with cover features for better quality reconstruction
        x = torch.cat([x, cover_features], dim=1)
        
        # Final processing
        container = self.final_conv(x)
        
        # Apply residual connection for cover preservation
        # This is key to improving PSNR/SSIM
        alpha = 0.5  # Balance parameter (can be made trainable)
        container = alpha * cover_img + (1 - alpha) * container
        
        return container

# Enhanced Extraction Network
class EnhancedExtractionNetwork(nn.Module):
    """
    Improved extraction network with higher capacity for better secret recovery
    """
    def __init__(self, img_size=144, window_size=8, embed_dim=128, depths=[6, 6, 6, 6],
                 num_heads=[8, 8, 8, 8], mlp_ratio=4.):
        super().__init__()
        
        # Initial convolutional embedding with enhanced architecture
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # RSTB blocks
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(img_size, img_size),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio
            )
            self.layers.append(layer)
        
        # Final processing with increased capacity
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, container_img):
        # Initial feature extraction
        x = self.initial_conv(container_img)
        
        # Process through RSTB blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final processing
        secret = self.final_conv(x)
        
        return secret

# Complete Enhanced Steganography System
class EnhancedSteganographySystem(nn.Module):
    """
    Complete end-to-end steganography system that compares embedding in
    high vs low attention areas and evaluates results
    """
    def __init__(self, img_size=144, embed_dim=128, depths=[6, 6, 6, 6], 
                 num_heads=[8, 8, 8, 8], window_size=8):
        super().__init__()
        
        # Cover and secret image attention generators
        self.cover_attention = DualAttentionHeatmapGenerator(
            dim=embed_dim//2, num_heads=num_heads[0]//2, 
            window_size=window_size, img_size=img_size
        )
        
        self.secret_attention = DualAttentionHeatmapGenerator(
            dim=embed_dim//2, num_heads=num_heads[0]//2, 
            window_size=window_size, img_size=img_size
        )
        
        # Hiding network
        self.hiding_network = EnhancedHidingNetwork(
            img_size=img_size, window_size=window_size, 
            embed_dim=embed_dim, depths=depths, num_heads=num_heads
        )
        
        # Extraction network
        self.extraction_network = EnhancedExtractionNetwork(
            img_size=img_size, window_size=window_size,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads
        )
    
    def compute_embedding_map(self, cover_attention, secret_attention, use_high_attention=True):
        """
        Compute optimal embedding map based on both attention maps
        """
        if use_high_attention:
            # Embed in regions where cover has high attention AND secret has significant content
            embedding_strength = cover_attention * secret_attention
        else:
            # Embed in regions where cover has low attention BUT secret has significant content
            embedding_strength = (1 - cover_attention) * secret_attention
            
        # Apply adaptive normalization for better balance
        embedding_strength = self.normalize_embedding_strength(embedding_strength)
        
        return embedding_strength
        
    def normalize_embedding_strength(self, strength_map):
        """
        Adaptive scaling to ensure enough embedding capacity
        while maintaining higher visual quality
        """
        min_strength = 0.4  # Minimum embedding strength 
        max_strength = 0.8  # Maximum embedding strength
        
        # Normalize to use full range while preserving relative values
        B, C, H, W = strength_map.shape
        flat_map = strength_map.view(B, -1)
        
        # Get min and max for each batch element
        min_vals = flat_map.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_vals = flat_map.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        
        # Normalize between min_strength and max_strength
        normalized = min_strength + (max_strength - min_strength) * (
            (strength_map - min_vals) / (max_vals - min_vals + 1e-8)
        )
        
        return normalized
    
    def forward_hide(self, cover_img, secret_img, use_high_attention=True):
        """
        Forward pass for hiding process
        """
        # Generate attention maps for both images
        cover_attention = self.cover_attention(cover_img)
        secret_attention = self.secret_attention(secret_img)
        
        # Compute embedding map
        embedding_map = self.compute_embedding_map(
            cover_attention, secret_attention, use_high_attention
        )
        
        # Generate container image
        container = self.hiding_network(cover_img, secret_img, embedding_map)
        
        return container, cover_attention, secret_attention, embedding_map
    
    def forward_extract(self, container):
        """
        Forward pass for extraction process
        """
        return self.extraction_network(container)
    
    def compare_attention_methods(self, cover_imgs, secret_imgs):
        """
        Compare high vs low attention methods and compute metrics
        """
        # Generate container images using both methods
        container_high, cover_atn, secret_atn, embed_map_high = self.forward_hide(
            cover_imgs, secret_imgs, use_high_attention=True
        )
        
        container_low, _, _, embed_map_low = self.forward_hide(
            cover_imgs, secret_imgs, use_high_attention=False
        )
        
        # Extract secret images from both containers
        extracted_high = self.forward_extract(container_high)
        extracted_low = self.forward_extract(container_low)
        
        # Calculate metrics
        metrics = {"high_attention": {"container": {}, "extracted": {}},
                  "low_attention": {"container": {}, "extracted": {}}}
        
        # Calculate average metrics across the batch
        batch_size = cover_imgs.size(0)
        for i in range(batch_size):
            # Get single images
            cover_np = cover_imgs[i].cpu().detach().permute(1, 2, 0).numpy()
            secret_np = secret_imgs[i].cpu().detach().permute(1, 2, 0).numpy()
            container_high_np = container_high[i].cpu().detach().permute(1, 2, 0).numpy()
            container_low_np = container_low[i].cpu().detach().permute(1, 2, 0).numpy()
            extracted_high_np = extracted_high[i].cpu().detach().permute(1, 2, 0).numpy()
            extracted_low_np = extracted_low[i].cpu().detach().permute(1, 2, 0).numpy()
            
            # Calculate metrics for high attention
            if i == 0:
                metrics["high_attention"]["container"]["psnr"] = calculate_psnr(cover_np, container_high_np)
                metrics["high_attention"]["container"]["ssim"] = calculate_ssim(cover_np, container_high_np)
                metrics["high_attention"]["extracted"]["psnr"] = calculate_psnr(secret_np, extracted_high_np)
                metrics["high_attention"]["extracted"]["ssim"] = calculate_ssim(secret_np, extracted_high_np)
                
                metrics["low_attention"]["container"]["psnr"] = calculate_psnr(cover_np, container_low_np)
                metrics["low_attention"]["container"]["ssim"] = calculate_ssim(cover_np, container_low_np)
                metrics["low_attention"]["extracted"]["psnr"] = calculate_psnr(secret_np, extracted_low_np)
                metrics["low_attention"]["extracted"]["ssim"] = calculate_ssim(secret_np, extracted_low_np)
            else:
                metrics["high_attention"]["container"]["psnr"] += calculate_psnr(cover_np, container_high_np)
                metrics["high_attention"]["container"]["ssim"] += calculate_ssim(cover_np, container_high_np)
                metrics["high_attention"]["extracted"]["psnr"] += calculate_psnr(secret_np, extracted_high_np)
                metrics["high_attention"]["extracted"]["ssim"] += calculate_ssim(secret_np, extracted_high_np)
                
                metrics["low_attention"]["container"]["psnr"] += calculate_psnr(cover_np, container_low_np)
                metrics["low_attention"]["container"]["ssim"] += calculate_ssim(cover_np, container_low_np)
                metrics["low_attention"]["extracted"]["psnr"] += calculate_psnr(secret_np, extracted_low_np)
                metrics["low_attention"]["extracted"]["ssim"] += calculate_ssim(secret_np, extracted_low_np)
        
        # Calculate averages
        metrics["high_attention"]["container"]["psnr"] /= batch_size
        metrics["high_attention"]["container"]["ssim"] /= batch_size
        metrics["high_attention"]["extracted"]["psnr"] /= batch_size
        metrics["high_attention"]["extracted"]["ssim"] /= batch_size
        
        metrics["low_attention"]["container"]["psnr"] /= batch_size
        metrics["low_attention"]["container"]["ssim"] /= batch_size
        metrics["low_attention"]["extracted"]["psnr"] /= batch_size
        metrics["low_attention"]["extracted"]["ssim"] /= batch_size
        
        # Create comparison visualization for the first sample in the batch
        cover_np = cover_imgs[0].cpu().detach().permute(1, 2, 0).numpy()
        secret_np = secret_imgs[0].cpu().detach().permute(1, 2, 0).numpy()
        container_high_np = container_high[0].cpu().detach().permute(1, 2, 0).numpy()
        container_low_np = container_low[0].cpu().detach().permute(1, 2, 0).numpy()
        extracted_high_np = extracted_high[0].cpu().detach().permute(1, 2, 0).numpy()
        extracted_low_np = extracted_low[0].cpu().detach().permute(1, 2, 0).numpy()
        cover_atn_np = cover_atn[0].cpu().detach().squeeze().numpy()
        secret_atn_np = secret_atn[0].cpu().detach().squeeze().numpy()
        embed_high_np = embed_map_high[0].cpu().detach().squeeze().numpy()
        embed_low_np = embed_map_low[0].cpu().detach().squeeze().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
        # Row 1: High attention
        axes[0, 0].imshow(cover_np)
        axes[0, 0].set_title("Cover Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(secret_np)
        axes[0, 1].set_title("Secret Image")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cover_atn_np, cmap='hot')
        axes[0, 2].set_title("Cover Attention")
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(container_high_np)
        axes[0, 3].set_title(f"Container (High Attention)\nPSNR: {metrics['high_attention']['container']['psnr']:.2f}dB\nSSIM: {metrics['high_attention']['container']['ssim']:.4f}")
        axes[0, 3].axis('off')
        
        axes[0, 4].imshow(extracted_high_np)
        axes[0, 4].set_title(f"Extracted (High Attention)\nPSNR: {metrics['high_attention']['extracted']['psnr']:.2f}dB\nSSIM: {metrics['high_attention']['extracted']['ssim']:.4f}")
        axes[0, 4].axis('off')
        
        # Row 2: Low attention
        axes[1, 0].imshow(cover_np)
        axes[1, 0].set_title("Cover Image")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(secret_np)
        axes[1, 1].set_title("Secret Image")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(secret_atn_np, cmap='hot')
        axes[1, 2].set_title("Secret Attention")
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(container_low_np)
        axes[1, 3].set_title(f"Container (Low Attention)\nPSNR: {metrics['low_attention']['container']['psnr']:.2f}dB\nSSIM: {metrics['low_attention']['container']['ssim']:.4f}")
        axes[1, 3].axis('off')
        
        axes[1, 4].imshow(extracted_low_np)
        axes[1, 4].set_title(f"Extracted (Low Attention)\nPSNR: {metrics['low_attention']['extracted']['psnr']:.2f}dB\nSSIM: {metrics['low_attention']['extracted']['ssim']:.4f}")
        axes[1, 4].axis('off')
        
        plt.tight_layout()
        
        # Return results dictionary
        return {
            "metrics": metrics,
            "visualization": fig,
            "container_high": container_high,
            "container_low": container_low,
            "extracted_high": extracted_high,
            "extracted_low": extracted_low,
            "cover_attention": cover_atn,
            "secret_attention": secret_atn,
            "embedding_map_high": embed_map_high,
            "embedding_map_low": embed_map_low
        }

# Legacy models kept for backward compatibility
class AttentionHeatmapGenerator(nn.Module):
    def __init__(self, dim=64, num_heads=6, window_size=8, img_size=144):
        super().__init__()
        self.dim = dim
        self.img_size = img_size

        # Initial feature extraction
        self.init_conv = nn.Conv2d(3, dim, kernel_size=3, padding=1)

        # Define input resolution properly
        self.input_resolution = (img_size, img_size)

        # Use proper Swin Transformer block
        self.swin_block = SwinTransformerBlock(
            dim=dim,
            input_resolution=self.input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2
        )

        # Add normalization layers
        self.norm = nn.LayerNorm(dim)

        # Process attention into heatmap 
        self.conv_process = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim//2, dim//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim//4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Ensure input dimensions match what we initialized with
        assert H == self.img_size and W == self.img_size, f"Input size ({H}x{W}) doesn't match expected size ({self.img_size}x{self.img_size})"

        # Extract features
        features = self.init_conv(x)  # B, dim, H, W

        # Reshape for Swin Transformer
        features_reshaped = features.flatten(2).transpose(1, 2)  # B, H*W, dim

        # Apply Swin Transformer
        transformed = self.swin_block(features_reshaped)

        # Normalize
        transformed = self.norm(transformed)

        # Reshape back to spatial form
        transformed = transformed.view(B, H, W, -1).permute(0, 3, 1, 2)  # B, dim, H, W

        # Process into attention heatmap
        heatmap = self.conv_process(transformed)

        return heatmap

class AttentionGuidedHidingNetwork(nn.Module):
    def __init__(self, img_size=144, window_size=8, embed_dim=128, depths=[2, 2, 2, 2],
                 num_heads=[8, 8, 8, 8], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
                     
        # Add attention heatmap generator
        self.heatmap_generator = AttentionHeatmapGenerator(dim=embed_dim, num_heads=num_heads[0], window_size=window_size, img_size=img_size)

        # Shallow Information Hiding module
        self.shallow_conv = nn.Conv2d(6, 64, kernel_size=3, padding=1)  # 3 (cover) + 3 (secret) = 6 channels

        # Deep Information Hiding module
        self.patch_embed = nn.Conv2d(64, embed_dim, kernel_size=3, padding=1)

        # RSTB blocks
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(img_size, img_size),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=None  # No downsampling in this architecture
            )
            self.layers.append(layer)

        self.norm = norm_layer(embed_dim)

        # Convert back to image space
        self.conv_out = nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1)

        # Construction Container Image module
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, cover_img, secret_img, use_high_attention=True):
        # Generate attention heatmap from cover image
        attention_map = self.heatmap_generator(cover_img)

        if use_high_attention:
        # Scale attention to keep even low values above a minimum threshold
           attention_map = 0.3 + (0.7 * attention_map)  # Ensures minimum value of 0.3
        else:
        # Invert and scale
           attention_map = 0.3 + (0.7 * (1 - attention_map))

        # Prepare secret image with attention weighting
        secret_expanded = secret_img  # Both are B, 3, H, W

        # Apply attention to secret image (multiply each channel)
        attention_expanded = attention_map.expand(-1, 3, -1, -1)  # Expand to B, 3, H, W
        weighted_secret = secret_expanded * attention_expanded

        # Combine cover and weighted secret images
        x = torch.cat([cover_img, weighted_secret], dim=1)

        # Continue with normal processing
        x = self.shallow_conv(x)
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.conv_out(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x, attention_map

class ExtractionNetwork(nn.Module):
    def __init__(self, img_size=144, window_size=8, embed_dim=96, depths=[6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()

        # Shallow Information Extraction module
        self.shallow_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 3 channels (container image)

        # Deep Information Extraction module
        self.patch_embed = nn.Conv2d(64, embed_dim, kernel_size=3, padding=1)

        # RSTB blocks
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(img_size, img_size),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=None  # No downsampling in this architecture
            )
            self.layers.append(layer)

        self.norm = norm_layer(embed_dim)

        # Convert back to image space
        self.conv_out = nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1)

        # Reconstruct Secret Image module
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, container_img):
        # Shallow information extraction
        x = self.shallow_conv(container_img)

        # Patch embedding
        x = self.patch_embed(x)

        # Apply RSTB blocks
        for layer in self.layers:
            x = layer(x)

        # Final processing
        x = self.conv_out(x)
        x = self.final_conv(x)

        # Apply sigmoid to keep values in [0,1]
        x = self.sigmoid(x)

        return x

# Function to run enhanced steganography pipeline
def run_enhanced_steganography_pipeline(cover_path, secret_path, output_dir="./output",
                                       model_weights=None, img_size=144):
    """
    Run the enhanced steganography pipeline that compares high vs low attention
    methods on the given images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess images
    cover_img = cv2.imread(cover_path)
    secret_img = cv2.imread(secret_path)
    
    # Resize to desired dimensions
    cover_img = cv2.resize(cover_img, (img_size, img_size))
    secret_img = cv2.resize(secret_img, (img_size, img_size))
    
    # Convert to RGB
    cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)
    secret_img = cv2.cvtColor(secret_img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0,1]
    cover_img = cover_img.astype(np.float32) / 255.0
    secret_img = secret_img.astype(np.float32) / 255.0
    
    # Convert to torch tensors
    cover_tensor = torch.from_numpy(cover_img).permute(2, 0, 1).unsqueeze(0)
    secret_tensor = torch.from_numpy(secret_img).permute(2, 0, 1).unsqueeze(0)
    
    # Create model
    model = EnhancedSteganographySystem(
        img_size=img_size, 
        embed_dim=128,
        depths=[6, 6, 6, 6],
        num_heads=[8, 8, 8, 8],
        window_size=8
    )
    
    # Load weights if provided
    if model_weights and os.path.exists(model_weights):
        model.load_state_dict(torch.load(model_weights))
        print(f"Loaded model weights from {model_weights}")
    
    # Set to evaluation mode
    model.eval()
    
    # Compare high vs low attention methods
    with torch.no_grad():
        results = model.compare_attention_methods(cover_tensor, secret_tensor)
    
    # Save visualization
    results["visualization"].savefig(os.path.join(output_dir, "comparison_results.jpg"))
    
    # Save individual images
    output_images = {
        "container_high": os.path.join(output_dir, "container_high.png"),
        "container_low": os.path.join(output_dir, "container_low.png"),
        "extracted_high": os.path.join(output_dir, "extracted_high.png"),
        "extracted_low": os.path.join(output_dir, "extracted_low.png"),
        "cover_attention": os.path.join(output_dir, "cover_attention.png"),
        "secret_attention": os.path.join(output_dir, "secret_attention.png"),
        "embedding_map_high": os.path.join(output_dir, "embedding_map_high.png"),
        "embedding_map_low": os.path.join(output_dir, "embedding_map_low.png")
    }
    
    # Save individual images
    for key, path in output_images.items():
        if key in results:
            # Convert tensor to numpy and save
            if key.startswith("container") or key.startswith("extracted"):
                img = results[key].squeeze(0).permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, img)
            else:
                # For attention maps
                img = results[key].squeeze().cpu().numpy()
                plt.figure(figsize=(6, 6))
                plt.imshow(img, cmap='hot')
                plt.colorbar()
                plt.title(key.replace('_', ' ').title())
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(path)
                plt.close()
    
    # Calculate detailed metrics between original and final images
    metrics = results["metrics"]
    
    print("\nHigh Attention Embedding Results:")
    print(f"Container Quality - PSNR: {metrics['high_attention']['container']['psnr']:.2f}dB, SSIM: {metrics['high_attention']['container']['ssim']:.4f}")
    print(f"Secret Recovery - PSNR: {metrics['high_attention']['extracted']['psnr']:.2f}dB, SSIM: {metrics['high_attention']['extracted']['ssim']:.4f}")
    
    print("\nLow Attention Embedding Results:")
    print(f"Container Quality - PSNR: {metrics['low_attention']['container']['psnr']:.2f}dB, SSIM: {metrics['low_attention']['container']['ssim']:.4f}")
    print(f"Secret Recovery - PSNR: {metrics['low_attention']['extracted']['psnr']:.2f}dB, SSIM: {metrics['low_attention']['extracted']['ssim']:.4f}")
    
    return {
        "metrics": metrics,
        "output_dir": output_dir,
        "comparison_path": os.path.join(output_dir, "comparison_results.jpg")
    }

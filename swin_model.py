import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt

# Utility Functions for Image Metrics
def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def calculate_ssim(original, compressed):
    return ssim(original, compressed, multichannel=True, channel_axis=2)

def calculate_mse(original, compressed):
    return np.mean((original - compressed) ** 2)

# Helper Functions
def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

# this function is used for partitioning the tensors into non overlapping windows
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)

        B : Batch size
        H : Height of the input feature map
        W : Width of the input featue map
        C : Number of Channels

        window_size (int): window size = The size of each square window
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C) # this is for dividing the windows
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)  # windows rearrangement
    return windows  # returns the divided windows

#this function does the opposite job of the windows_partition function
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

# Mlp module
# Multi- Layer Perceptron Class - this class defines a simple feedforward neural network module for the Swin Transformer,
# It is a two layer FFN
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features) # fully connected layer 1
        self.act = act_layer()   # activation layer
        self.fc2 = nn.Linear(hidden_features, out_features) # fully connected layer 2
        self.drop = nn.Dropout(drop)  # drop out layer

    def forward(self, x):
        x = self.fc1(x)  # apply Linear Transformation 1
        x = self.act(x)  # activation function
        x = self.drop(x) # apply Drop out layer
        x = self.fc2(x)  # apply Linear Transformation 2
        x = self.drop(x) # apply Drop out layer
        return x

# Window Attention Module
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
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # 2, Wh, Ww # 2, Wh, Ww
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

# Patch Unembedding (for converting features back to image)
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

# Residual Swin-Transformer Block (RSTB) as per the paper
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
            shift_size=window_size // 2  # Add proper shift size
        )

        # Add normalization layers
        self.norm = nn.LayerNorm(dim)

        # Process attention into heatmap with better design
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

        # Reshape back to spatial form - this is the key fix
        transformed = transformed.view(B, H, W, -1).permute(0, 3, 1, 2)  # B, dim, H, W

        # Process into attention heatmap
        heatmap = self.conv_process(transformed)

        return heatmap


'''
# Hiding Network as per the paper's architecture
class HidingNetwork(nn.Module):
    def __init__(self, img_size=144, window_size=8, embed_dim=96, depths=[6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()

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

    def forward(self, cover_img, secret_img):
        # Combine cover and secret images
        x = torch.cat([cover_img, secret_img], dim=1)

        # Shallow information hiding
        x = self.shallow_conv(x)

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
'''
# Modify the HidingNetwork to include attention-guided embedding
class AttentionGuidedHidingNetwork(nn.Module):
    def __init__(self, img_size=144, window_size=8, embed_dim=128, depths=[2, 2, 2, 2],
                 num_heads=[8, 8, 8, 8], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()

        # Add attention heatmap generator
        self.heatmap_generator = AttentionHeatmapGenerator(dim=embed_dim, num_heads=num_heads[0], window_size=window_size)

        # Rest of the hiding network remains the same
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

        # Invert heatmap if using low-attention regions
        '''
        if not use_high_attention:
            attention_map = 1 - attention_map
        '''
        if use_high_attention:
        # Scale attention to keep even low values above a minimum threshold
           attention_map = 0.3 + (0.7 * attention_map)  # Ensures minimum value of 0.3

        else:
        # Invert and scale
           attention_map = 0.3 + (0.7 * (1 - attention_map))

        # Prepare secret image with attention weighting
        # Convert to same dimensions for multiplication
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


 # Extraction Network as per the paper's architecture
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

'''
# Main function to process images
def process_images(cover_path, secret_path, output_path):
    # Read images
    cover_img = cv2.imread(cover_path)
    secret_img = cv2.imread(secret_path)

    # Resize to 144x144 as per the paper
    cover_img = cv2.resize(cover_img, (144, 144))
    secret_img = cv2.resize(secret_img, (144, 144))

    # Convert to RGB if in BGR
    cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)
    secret_img = cv2.cvtColor(secret_img, cv2.COLOR_BGR2RGB)

    # Normalize to [0,1]
    cover_img = cover_img.astype(np.float32) / 255.0
    secret_img = secret_img.astype(np.float32) / 255.0

    # Convert to torch tensors
    cover_tensor = torch.from_numpy(cover_img).permute(2, 0, 1).unsqueeze(0)
    secret_tensor = torch.from_numpy(secret_img).permute(2, 0, 1).unsqueeze(0)

    # Create model
    model = HidingNetwork(img_size=144, window_size=8, embed_dim=96,
                         depths=[2, 2, 2, 2], num_heads=[6, 6, 6, 6])

    # Since we're not training, set to eval mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        container_tensor = model(cover_tensor, secret_tensor)

    # Convert back to numpy for saving and metrics calculation
    container_img = container_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
    container_img = container_img.astype(np.uint8)

    # Convert back to BGR for saving with OpenCV
    container_img_bgr = cv2.cvtColor(container_img, cv2.COLOR_RGB2BGR)

    # Save the container image
    cv2.imwrite(output_path, container_img_bgr)

    # Calculate metrics
    cover_img_uint8 = (cover_img * 255.0).astype(np.uint8)
    container_img_rgb = cv2.cvtColor(container_img_bgr, cv2.COLOR_BGR2RGB)

    psnr_value = calculate_psnr(cover_img_uint8, container_img_rgb)
    ssim_value = calculate_ssim(cover_img_uint8, container_img_rgb)
    mse_value = calculate_mse(cover_img_uint8, container_img_rgb)

    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"MSE: {mse_value:.2f}")

    return {
        "psnr": psnr_value,
        "ssim": ssim_value,
        "mse": mse_value,
        "container_path": output_path
    }

'''


# New function to process images with attention guidance
def process_images_with_attention(cover_path, secret_path, output_high_path, output_low_path, attention_map_path=None):
    # Read images
    cover_img = cv2.imread(cover_path)
    secret_img = cv2.imread(secret_path)

    # Resize to 144x144 as per the paper
    cover_img = cv2.resize(cover_img, (144, 144))
    secret_img = cv2.resize(secret_img, (144, 144))

    # Convert to RGB if in BGR
    cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)
    secret_img = cv2.cvtColor(secret_img, cv2.COLOR_BGR2RGB)

    # Normalize to [0,1]
    cover_img = cover_img.astype(np.float32) / 255.0
    secret_img = secret_img.astype(np.float32) / 255.0

    # Convert to torch tensors
    cover_tensor = torch.from_numpy(cover_img).permute(2, 0, 1).unsqueeze(0)
    secret_tensor = torch.from_numpy(secret_img).permute(2, 0, 1).unsqueeze(0)

    # Create model
    model = AttentionGuidedHidingNetwork(img_size=144, window_size=8, embed_dim=96,
                                         depths=[2, 2, 2, 2], num_heads=[6, 6, 6, 6])

    # Set to eval mode
    model.eval()

    # Forward pass with high attention regions
    with torch.no_grad():
        container_high_tensor, attention_map_high = model(cover_tensor, secret_tensor, use_high_attention=True)
        container_low_tensor, attention_map_low = model(cover_tensor, secret_tensor, use_high_attention=False)

    # Convert tensors back to numpy
    container_high_img = container_high_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
    container_high_img = container_high_img.astype(np.uint8)

    container_low_img = container_low_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
    container_low_img = container_low_img.astype(np.uint8)

    # Convert to BGR for saving with OpenCV
    container_high_bgr = cv2.cvtColor(container_high_img, cv2.COLOR_RGB2BGR)
    container_low_bgr = cv2.cvtColor(container_low_img, cv2.COLOR_RGB2BGR)

    # Save the container images
    cv2.imwrite(output_high_path, container_high_bgr)
    cv2.imwrite(output_low_path, container_low_bgr)

    # Save attention maps if requested
    if attention_map_path:
        # Convert attention maps to heatmap visualization
        attention_high_np = attention_map_high.squeeze().cpu().numpy()
        attention_low_np = attention_map_low.squeeze().cpu().numpy()

        # Create a figure with both attention maps
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(attention_high_np, cmap='hot')
        plt.title('High Attention Regions')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(attention_low_np, cmap='hot')
        plt.title('Low Attention Regions')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(attention_map_path)

    # Calculate metrics for high attention embedding
    cover_img_uint8 = (cover_img * 255.0).astype(np.uint8)
    container_high_rgb = cv2.cvtColor(container_high_bgr, cv2.COLOR_BGR2RGB)

    psnr_high = calculate_psnr(cover_img_uint8, container_high_rgb)
    ssim_high = calculate_ssim(cover_img_uint8, container_high_rgb)
    mse_high = calculate_mse(cover_img_uint8, container_high_rgb)

    # Calculate metrics for low attention embedding
    container_low_rgb = cv2.cvtColor(container_low_bgr, cv2.COLOR_BGR2RGB)

    psnr_low = calculate_psnr(cover_img_uint8, container_low_rgb)
    ssim_low = calculate_ssim(cover_img_uint8, container_low_rgb)
    mse_low = calculate_mse(cover_img_uint8, container_low_rgb)

    print(f"High Attention Embedding - PSNR: {psnr_high:.2f} dB, SSIM: {ssim_high:.4f}, MSE: {mse_high:.2f}")
    print(f"Low Attention Embedding - PSNR: {psnr_low:.2f} dB, SSIM: {ssim_low:.4f}, MSE: {mse_low:.2f}")

    return {
        "high_attention": {
            "psnr": psnr_high,
            "ssim": ssim_high,
            "mse": mse_high,
            "container_path": output_high_path
        },
        "low_attention": {
            "psnr": psnr_low,
            "ssim": ssim_low,
            "mse": mse_low,
            "container_path": output_low_path
        },
        "attention_map_path": attention_map_path
    }



'''
# Function to extract secret image from container image
def extract_secret_image(container_path, output_path):
    # Read container image
    container_img = cv2.imread(container_path)

    # Resize to 144x144 if needed
    container_img = cv2.resize(container_img, (144, 144))

    # Convert to RGB
    container_img = cv2.cvtColor(container_img, cv2.COLOR_BGR2RGB)

    # Normalize to [0,1]
    container_img = container_img.astype(np.float32) / 255.0

    # Convert to torch tensor
    container_tensor = torch.from_numpy(container_img).permute(2, 0, 1).unsqueeze(0)

    # Create extraction model
    model = ExtractionNetwork(img_size=144, window_size=8, embed_dim=96,
                             depths=[2, 2, 2, 2], num_heads=[6, 6, 6, 6])

    # Set to eval mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        extracted_secret_tensor = model(container_tensor)

    # Convert back to numpy for saving
    extracted_secret_img = extracted_secret_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
    extracted_secret_img = extracted_secret_img.astype(np.uint8)

    # Convert to BGR for saving with OpenCV
    extracted_secret_img_bgr = cv2.cvtColor(extracted_secret_img, cv2.COLOR_RGB2BGR)

    # Save the extracted secret image
    cv2.imwrite(output_path, extracted_secret_img_bgr)

    return {
        "extracted_secret_path": output_path
    }
'''

# Modify the extraction function to extract from attention-guided stego images
def extract_secret_image_with_attention(container_path, output_path):
    # Read container image
    container_img = cv2.imread(container_path)

    # Resize to 144x144 if needed
    container_img = cv2.resize(container_img, (144, 144))

    # Convert to RGB
    container_img = cv2.cvtColor(container_img, cv2.COLOR_BGR2RGB)

    # Normalize to [0,1]
    container_img = container_img.astype(np.float32) / 255.0

    # Convert to torch tensor
    container_tensor = torch.from_numpy(container_img).permute(2, 0, 1).unsqueeze(0)

    # Create extraction model - same as before, no changes needed
    model = ExtractionNetwork(img_size=144, window_size=8, embed_dim=96,
                              depths=[2, 2, 2, 2], num_heads=[6, 6, 6, 6])

    # Set to eval mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        extracted_secret_tensor = model(container_tensor)

    # Convert back to numpy for saving
    extracted_secret_img = extracted_secret_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
    extracted_secret_img = extracted_secret_img.astype(np.uint8)

    # Convert to BGR for saving with OpenCV
    extracted_secret_img_bgr = cv2.cvtColor(extracted_secret_img, cv2.COLOR_RGB2BGR)

    # Save the extracted secret image
    cv2.imwrite(output_path, extracted_secret_img_bgr)

    return {
        "extracted_secret_path": output_path
    }

'''

# Complete pipeline function
def steganography_pipeline(cover_path, secret_path, container_output_path, extracted_secret_output_path):
    # First, hide the secret image
    hide_metrics = process_images(cover_path, secret_path, container_output_path)  # calling the process_images function for processing the images

    # Then, extract the secret image
    extract_metrics = extract_secret_image(container_output_path, extracted_secret_output_path)  # secret image extraction

    # Load all images for visualization
    cover_img = cv2.imread(cover_path)
    secret_img = cv2.imread(secret_path)
    container_img = cv2.imread(container_output_path)
    extracted_img = cv2.imread(extracted_secret_output_path)

    # Convert BGR to RGB for correct display
    cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)
    secret_img = cv2.cvtColor(secret_img, cv2.COLOR_BGR2RGB)
    container_img = cv2.cvtColor(container_img, cv2.COLOR_BGR2RGB)
    extracted_img = cv2.cvtColor(extracted_img, cv2.COLOR_BGR2RGB)

    # Calculate metrics for secret vs extracted
    secret_img_resized = cv2.resize(secret_img, (144, 144))
    extracted_img_resized = cv2.resize(extracted_img, (144, 144))

    # Calculate metrics
    psnr_secret = calculate_psnr(secret_img_resized, extracted_img_resized)
    ssim_secret = calculate_ssim(secret_img_resized, extracted_img_resized)
    mse_secret = calculate_mse(secret_img_resized, extracted_img_resized)

    # Print results
    print(f"Container image saved to: {hide_metrics['container_path']}")
    print(f"Extracted secret image saved to: {extract_metrics['extracted_secret_path']}")
    print(f"Hiding Metrics: PSNR={hide_metrics['psnr']:.2f}dB, SSIM={hide_metrics['ssim']:.4f}, MSE={hide_metrics['mse']:.2f}")
    print(f"Extraction Metrics: PSNR={psnr_secret:.2f}dB, SSIM={ssim_secret:.4f}, MSE={mse_secret:.2f}")

    # Display images
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(cover_img)
    axes[0, 0].set_title("Cover Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(secret_img)
    axes[0, 1].set_title("Secret Image")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(container_img)
    axes[1, 0].set_title("Container Image")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(extracted_img)
    axes[1, 1].set_title("Extracted Secret Image")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

    return {
        "hide_metrics": hide_metrics,
        "extract_metrics": {
            "psnr": psnr_secret,
            "ssim": ssim_secret,
            "mse": mse_secret,
            "extracted_secret_path": extract_metrics["extracted_secret_path"]
        }
    }

'''

# Complete pipeline function for attention-guided steganography
def attention_guided_steganography_pipeline(cover_path, secret_path, output_dir="./output"):
  
    os.makedirs(output_dir, exist_ok=True)

    # Define output paths
    container_high_path = os.path.join(output_dir, "container_high_attention.jpg")
    container_low_path = os.path.join(output_dir, "container_low_attention.jpg")
    attention_map_path = os.path.join(output_dir, "attention_maps.jpg")
    extracted_high_path = os.path.join(output_dir, "extracted_high_attention.jpg")
    extracted_low_path = os.path.join(output_dir, "extracted_low_attention.jpg")

    # Process images with both high and low attention guidance
    hide_metrics = process_images_with_attention(
        cover_path, secret_path,
        container_high_path, container_low_path,
        attention_map_path
    )

    # Extract secret images from both container images
    extract_high_metrics = extract_secret_image_with_attention(container_high_path, extracted_high_path)
    extract_low_metrics = extract_secret_image_with_attention(container_low_path, extracted_low_path)

    # Load all images for visualization
    cover_img = cv2.imread(cover_path)
    secret_img = cv2.imread(secret_path)
    container_high_img = cv2.imread(container_high_path)
    container_low_img = cv2.imread(container_low_path)
    extracted_high_img = cv2.imread(extracted_high_path)
    extracted_low_img = cv2.imread(extracted_low_path)

    # Convert BGR to RGB for display
    cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)
    secret_img = cv2.cvtColor(secret_img, cv2.COLOR_BGR2RGB)
    container_high_img = cv2.cvtColor(container_high_img, cv2.COLOR_BGR2RGB)
    container_low_img = cv2.cvtColor(container_low_img, cv2.COLOR_BGR2RGB)
    extracted_high_img = cv2.cvtColor(extracted_high_img, cv2.COLOR_BGR2RGB)
    extracted_low_img = cv2.cvtColor(extracted_low_img, cv2.COLOR_BGR2RGB)

    # Resize all images to 144x144 for fair comparison
    cover_img = cv2.resize(cover_img, (144, 144))
    secret_img = cv2.resize(secret_img, (144, 144))
    container_high_img = cv2.resize(container_high_img, (144, 144))
    container_low_img = cv2.resize(container_low_img, (144, 144))
    extracted_high_img = cv2.resize(extracted_high_img, (144, 144))
    extracted_low_img = cv2.resize(extracted_low_img, (144, 144))

    # Calculate extraction metrics (secret vs extracted)
    psnr_high_extract = calculate_psnr(secret_img, extracted_high_img)
    ssim_high_extract = calculate_ssim(secret_img, extracted_high_img)
    mse_high_extract = calculate_mse(secret_img, extracted_high_img)

    psnr_low_extract = calculate_psnr(secret_img, extracted_low_img)
    ssim_low_extract = calculate_ssim(secret_img, extracted_low_img)
    mse_low_extract = calculate_mse(secret_img, extracted_low_img)

    # Print results
    print("\nHigh Attention Embedding Results:")
    print(f"Container Image - PSNR: {hide_metrics['high_attention']['psnr']:.2f}dB, SSIM: {hide_metrics['high_attention']['ssim']:.4f}")
    print(f"Extracted Secret - PSNR: {psnr_high_extract:.2f}dB, SSIM: {ssim_high_extract:.4f}")

    print("\nLow Attention Embedding Results:")
    print(f"Container Image - PSNR: {hide_metrics['low_attention']['psnr']:.2f}dB, SSIM: {hide_metrics['low_attention']['ssim']:.4f}")
    print(f"Extracted Secret - PSNR: {psnr_low_extract:.2f}dB, SSIM: {ssim_low_extract:.4f}")

    # Display all images
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    axes[0, 0].imshow(cover_img)
    axes[0, 0].set_title("Cover Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(secret_img)
    axes[0, 1].set_title("Secret Image")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(container_high_img)
    axes[1, 0].set_title(f"Container Image (High Attention)\nPSNR: {hide_metrics['high_attention']['psnr']:.2f}dB, SSIM: {hide_metrics['high_attention']['ssim']:.4f}")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(container_low_img)
    axes[1, 1].set_title(f"Container Image (Low Attention)\nPSNR: {hide_metrics['low_attention']['psnr']:.2f}dB, SSIM: {hide_metrics['low_attention']['ssim']:.4f}")
    axes[1, 1].axis("off")

    axes[2, 0].imshow(extracted_high_img)
    axes[2, 0].set_title(f"Extracted Secret (High Attention)\nPSNR: {psnr_high_extract:.2f}dB, SSIM: {ssim_high_extract:.4f}")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(extracted_low_img)
    axes[2, 1].set_title(f"Extracted Secret (Low Attention)\nPSNR: {psnr_low_extract:.2f}dB, SSIM: {ssim_low_extract:.4f}")
    axes[2, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_results.jpg"))
    plt.show()

    # Return all metrics for further analysis
    return {
        "high_attention": {
            "container": {
                "psnr": hide_metrics['high_attention']['psnr'],
                "ssim": hide_metrics['high_attention']['ssim'],
                "mse": hide_metrics['high_attention']['mse'],
                "path": container_high_path
            },
            "extracted": {
                "psnr": psnr_high_extract,
                "ssim": ssim_high_extract,
                "mse": mse_high_extract,
                "path": extracted_high_path
            }
        },
        "low_attention": {
            "container": {
                "psnr": hide_metrics['low_attention']['psnr'],
                "ssim": hide_metrics['low_attention']['ssim'],
                "mse": hide_metrics['low_attention']['mse'],
                "path": container_low_path
            },
            "extracted": {
                "psnr": psnr_low_extract,
                "ssim": ssim_low_extract,
                "mse": mse_low_extract,
                "path": extracted_low_path
            }
        },
        "attention_map_path": attention_map_path,
        "comparison_path": os.path.join(output_dir, "comparison_results.jpg")
    }


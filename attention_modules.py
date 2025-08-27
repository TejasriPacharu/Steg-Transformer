import torch
import torch.nn as nn
import torch.nn.functional as F
from base_modules import SwinTransformerBlock
from utils import to_2tuple

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
            nn.LeakyReLU(0.2),
            # Add additional layer for better feature extraction
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
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
            ),
            # Add a third block for more capacity
            SwinTransformerBlock(
                dim=dim,
                input_resolution=self.input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0
            )
        ])
        
        # Normalization layer
        self.norm = nn.LayerNorm(dim)
        
        # Edge detector for better structure-aware attention
        self.edge_detector = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Process attention into heatmap with enhanced design
        self.conv_process = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim//2, dim//4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim//4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Refinement module for the attention map
        self.refine = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # 1 from attention + 1 from edge
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Ensure input dimensions match
        assert H == self.img_size and W == self.img_size, f"Input size mismatch"

        # Extract features
        features = self.init_conv(x)  # B, dim, H, W

        # Detect edges for structure awareness
        edges = self.edge_detector(x)  # B, 1, H, W
        
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
        initial_heatmap = self.conv_process(features_spatial)
        
        # Combine with edge information for refined attention
        combined = torch.cat([initial_heatmap, edges], dim=1)
        refined_heatmap = self.refine(combined)
        
        # Create sharper attention using gamma correction and edge enhancement
        gamma = 0.7  # < 1 emphasizes mid-tone regions
        sharpened = torch.pow(refined_heatmap, gamma)
        
        # Normalize to ensure proper range
        min_vals = sharpened.view(B, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_vals = sharpened.view(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        normalized = (sharpened - min_vals) / (max_vals - min_vals + 1e-8)
        
        return normalized
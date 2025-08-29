import torch
import torch.nn as nn
import torch.nn.functional as F
from base_modules import SwinTransformerBlock
from utils import to_2tuple

class DualAttentionHeatmapGenerator(nn.Module):
    """
    Enhanced attention map generator that produces high-quality attention maps
    with improved detail preservation and gradient sensitivity
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
            # Replace sigmoid with tanh for better gradient flow
            nn.Tanh()
        )
        
        # Add gradient detector for texture details
        self.gradient_detector = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),  # Larger kernel to capture more context
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Process attention into heatmap with enhanced design
        self.conv_process = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim//2, dim//4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim//4, 1, kernel_size=1),
            # Remove sigmoid here to avoid binarizing features too early
            nn.LeakyReLU(0.2)
        )
        
        # Refinement module for the attention map
        self.refine = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 1 from attention + 1 from edge + 1 from gradient
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            # Use softer activation for more nuanced attention maps
            nn.Tanh()  # Output range -1 to 1
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Ensure input dimensions match
        assert H == self.img_size and W == self.img_size, f"Input size mismatch"

        # Extract features
        features = self.init_conv(x)  # B, dim, H, W

        # Detect edges for structure awareness - normalized to [-1, 1]
        edges = self.edge_detector(x)  # B, 1, H, W
        
        # Detect gradients for texture detail
        gradients = self.gradient_detector(x)  # B, 1, H, W
        
        # Reshape for Swin Transformer
        features_reshaped = features.flatten(2).transpose(1, 2)  # B, H*W, dim

        # Apply Swin Transformer Blocks sequentially
        for block in self.swin_blocks:
            features_reshaped = block(features_reshaped)

        # Normalize
        features_reshaped = self.norm(features_reshaped)

        # Reshape back to spatial form
        features_spatial = features_reshaped.view(B, H, W, -1).permute(0, 3, 1, 2)  # B, dim, H, W

        # Process into initial attention heatmap (without early sigmoid)
        initial_heatmap = self.conv_process(features_spatial)
        
        # Scale initial heatmap to positive range for refinement
        initial_heatmap = F.softplus(initial_heatmap)
        
        # Combine with edge and gradient information for refined attention
        combined = torch.cat([initial_heatmap, edges, gradients], dim=1)
        refined_heatmap = self.refine(combined)
        
        # Convert from [-1, 1] to [0, 1] for final attention map
        refined_heatmap = (refined_heatmap + 1) / 2
        
        # Apply histogram equalization-like stretching for better detail distribution
        # This ensures we use the full dynamic range without binarizing
        batch_maps = []
        for b in range(B):
            single_map = refined_heatmap[b, 0]  # Extract single attention map
            
            # Calculate percentiles instead of min/max to avoid outlier influence
            low_percentile = torch.quantile(single_map.flatten(), 0.05)
            high_percentile = torch.quantile(single_map.flatten(), 0.95)
            
            # Stretch the map based on percentiles
            stretched_map = (single_map - low_percentile) / (high_percentile - low_percentile + 1e-8)
            stretched_map = torch.clamp(stretched_map, 0, 1)
            
            # Apply adaptive gamma correction for better mid-tones
            mid_level = torch.mean(stretched_map)
            gamma = 0.8 if mid_level < 0.5 else 1.2  # Adapt gamma based on brightness
            
            corrected_map = torch.pow(stretched_map, gamma)
            batch_maps.append(corrected_map.unsqueeze(0).unsqueeze(0))
        
        # Combine batch results
        normalized_maps = torch.cat(batch_maps, dim=0)
        
        # Apply local contrast enhancement
        kernel_size = 5
        local_mean = F.avg_pool2d(normalized_maps, kernel_size, stride=1, padding=kernel_size//2)
        enhanced_maps = normalized_maps + 0.5 * (normalized_maps - local_mean)
        enhanced_maps = torch.clamp(enhanced_maps, 0.05, 0.95)  # Prevent complete black/white regions
        
        return enhanced_maps
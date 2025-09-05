import torch
import torch.nn as nn
import torch.nn.functional as F
from base_modules import SwinTransformerBlock
from utils import to_2tuple

# class DualAttentionHeatmapGenerator(nn.Module):
#     """
#     Enhanced attention map generator that produces high-quality attention maps
#     with improved detail preservation and gradient sensitivity
#     """
#     def __init__(self, dim=64, num_heads=6, window_size=8, img_size=144):
#         super().__init__()
#         self.dim = dim
#         self.img_size = img_size
#         self.input_resolution = (img_size, img_size)
        
#         # Initial feature extraction with deeper architecture and more channels
#         self.init_conv = nn.Sequential(
#             nn.Conv2d(3, dim, kernel_size=3, padding=1),  # Increased channels from dim//2 to dim
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             # Add additional layer for better feature extraction
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2)
#         )
        
#         # Multiple Swin Transformer blocks for better attention modeling
#         self.swin_blocks = nn.ModuleList([
#             SwinTransformerBlock(
#                 dim=dim,
#                 input_resolution=self.input_resolution,
#                 num_heads=num_heads,
#                 window_size=window_size,
#                 shift_size=0
#             ),
#             SwinTransformerBlock(
#                 dim=dim,
#                 input_resolution=self.input_resolution,
#                 num_heads=num_heads,
#                 window_size=window_size,
#                 shift_size=window_size // 2
#             ),
#             # Add a third block for more capacity
#             SwinTransformerBlock(
#                 dim=dim,
#                 input_resolution=self.input_resolution,
#                 num_heads=num_heads,
#                 window_size=window_size,
#                 shift_size=0
#             )
#         ])
        
#         # Normalization layer
#         self.norm = nn.LayerNorm(dim)
        
#         # Enhanced edge detector using Sobel filters
#         self.edge_detector = nn.Sequential(
#             # Horizontal Sobel filter
#             nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
#             nn.LeakyReLU(0.2),
#             # Vertical Sobel filter
#             nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
#             nn.LeakyReLU(0.2),
#             # Combine edge information
#             nn.Conv2d(8, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
        
#         # Initialize Sobel filters
#         with torch.no_grad():
#             # Horizontal Sobel kernel
#             self.edge_detector[0].weight.data[:4] = torch.tensor([
#                 [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#                 [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#                 [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#                 [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
#             ]).reshape(4, 1, 3, 3).repeat(1, 3, 1, 1)
            
#             # Vertical Sobel kernel
#             self.edge_detector[2].weight.data[:4] = torch.tensor([
#                 [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#                 [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#                 [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#                 [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
#             ]).reshape(4, 1, 3, 3).repeat(1, 8, 1, 1)  # Fixed repeat dimension from 2 to 8
        
#         # Improved gradient detector with multi-scale perception
#         self.gradient_detector = nn.Sequential(
#             # Multi-scale feature extraction
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(16, 16, kernel_size=5, padding=2),  # Larger kernel for texture context
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(16, 16, kernel_size=7, padding=3),  # Even larger kernel for more context
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(16, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
        
#         # Semantic feature extractor for content-aware attention
#         self.semantic_extractor = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
        
#         # Process attention into heatmap with enhanced design
#         self.conv_process = nn.Sequential(
#             nn.Conv2d(dim, dim//2, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(dim//2, dim//4, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(dim//4, 1, kernel_size=1),
#             nn.S
#         )
        
#         # # Refinement module for the attention map
#         # self.refine = nn.Sequential(
#         #     nn.Conv2d(4, 32, kernel_size=3, padding=1),  # 1 from attention + 1 from edge + 1 from gradient + 1 from semantic
#         #     nn.LeakyReLU(0.2),
#         #     nn.Conv2d(32, 16, kernel_size=3, padding=1),
#         #     nn.LeakyReLU(0.2),
#         #     nn.Conv2d(16, 1, kernel_size=3, padding=1),
#         #     nn.Sigmoid()
#         # )
        
#         # # Final contrast enhancement
#         # self.contrast_enhancer = nn.Sequential(
#         #     nn.Conv2d(1, 16, kernel_size=3, padding=1),
#         #     nn.LeakyReLU(0.2),
#         #     nn.Conv2d(16, 1, kernel_size=3, padding=1),
#         #     nn.Sigmoid()
#         # )

#     def forward(self, x):
#         B, C, H, W = x.shape
        
#         # Ensure input dimensions match
#         assert H == self.img_size and W == self.img_size, f"Input size mismatch"

#         # Extract features
#         features = self.init_conv(x)  # B, dim, H, W

#         # Detect edges for structure awareness - normalized to [0, 1]
#         edges = self.edge_detector(x)  # B, 1, H, W
        
#         # Detect gradients for texture detail
#         gradients = self.gradient_detector(x)  # B, 1, H, W
        
#         # Extract semantic features
#         semantic = self.semantic_extractor(x)  # B, 1, H, W
        
#         # Reshape for Swin Transformer
#         features_reshaped = features.flatten(2).transpose(1, 2)  # B, H*W, dim

#         # Apply Swin Transformer Blocks sequentially
#         for block in self.swin_blocks:
#             features_reshaped = block(features_reshaped)

#         # Normalize
#         features_reshaped = self.norm(features_reshaped)

#         # Reshape back to spatial form
#         features_spatial = features_reshaped.view(B, H, W, -1).permute(0, 3, 1, 2)  # B, dim, H, W

#         # Process into initial attention heatmap (with sigmoid)
#         initial_heatmap = self.conv_process(features_spatial)
        
#         # Combine with edge, gradient, and semantic information for refined attention
#         combined = torch.cat([initial_heatmap, edges, gradients, semantic], dim=1)
#         #refined_heatmap = self.refine(combined)
        
#         # Apply adaptive histogram equalization for better distribution
#         batch_maps = []
#         for b in range(B):
#             single_map = initial_heatmap[b, 0]  # Extract single attention map
            
#             # Calculate percentiles for robust normalization (1-99% range)
#             low_percentile = torch.quantile(single_map.flatten(), 0.01)
#             high_percentile = torch.quantile(single_map.flatten(), 0.99)
            
#             # Stretch the map based on percentiles
#             stretched_map = (single_map - low_percentile) / (high_percentile - low_percentile + 1e-8)
#             stretched_map = torch.clamp(stretched_map, 0.02, 0.98)  # Allow wider dynamic range
            
#             # Apply content-adaptive gamma correction
#             mean_value = torch.mean(stretched_map)
#             if mean_value > 0.5:
#                 gamma = 1.5  # Emphasize bright areas
#             elif mean_value < 0.3:
#                 gamma = 0.7  # Emphasize dark areas
#             else:
#                 gamma = 1.0  # No change for balanced maps
                
#             # Apply gamma correction
#             gamma_corrected = torch.pow(stretched_map, gamma)
            
#             # Apply local contrast enhancement using guided filter approach
#             kernel_size = 7
#             padding = kernel_size // 2
            
#             # Calculate local mean
#             local_mean = F.avg_pool2d(
#                 F.pad(gamma_corrected.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect'),
#                 kernel_size, stride=1
#             )
            
#             # Calculate local variance
#             local_var = F.avg_pool2d(
#                 F.pad(gamma_corrected.unsqueeze(0).unsqueeze(0) ** 2, (padding, padding, padding, padding), mode='reflect'),
#                 kernel_size, stride=1
#             ) - local_mean ** 2
            
#             # Guided filter parameters
#             eps = 1e-4
#             a = local_var / (local_var + eps)
#             b = (1 - a) * local_mean
            
#             # Apply guided filter
#             enhanced_map = a * gamma_corrected.unsqueeze(0).unsqueeze(0) + b
#             enhanced_map = enhanced_map.squeeze(0).squeeze(0)
            
#             # Ensure the map has appropriate contrast
#             enhanced_map = (enhanced_map - enhanced_map.min()) / (enhanced_map.max() - enhanced_map.min() + 1e-8)
            
#             batch_maps.append(enhanced_map.unsqueeze(0).unsqueeze(0))
        
#         # Combine batch results
#         normalized_maps = torch.cat(batch_maps, dim=0)
        
#         # Final contrast enhancement
#         #final_maps = self.contrast_enhancer(normalized_maps)
        
#         # Ensure maps have good contrast and clear focus areas
#         final_maps = torch.clamp(normalized_maps, 0.05, 0.95)
        
#         return final_maps


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
            # Replace tanh with sigmoid for proper attention mapping
            nn.Sigmoid()
        )
        
        # Add gradient detector for texture details
        self.gradient_detector = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),  # Larger kernel to capture more context
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Change to sigmoid for proper attention mapping
        )
        
        # Process attention into heatmap with enhanced design
        self.conv_process = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim//2, dim//4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim//4, 1, kernel_size=1),
            # Add sigmoid here to properly normalize feature maps
            nn.Sigmoid()
        )
        
        # Refinement module for the attention map
        self.refine = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 1 from attention + 1 from edge + 1 from gradient
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            # Use sigmoid for proper attention map normalization (0 to 1 range)
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Ensure input dimensions match
        assert H == self.img_size and W == self.img_size, f"Input size mismatch"

        # Extract features
        features = self.init_conv(x)  # B, dim, H, W

        # Detect edges for structure awareness - normalized to [0, 1]
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

        # Process into initial attention heatmap (with sigmoid)
        initial_heatmap = self.conv_process(features_spatial)
        
        # Combine with edge and gradient information for refined attention
        combined = torch.cat([initial_heatmap, edges, gradients], dim=1)
        refined_heatmap = self.refine(combined)
        
        # Apply adaptive histogram equalization for better distribution
        batch_maps = []
        for b in range(B):
            single_map = refined_heatmap[b, 0]  # Extract single attention map
            
            # Calculate percentiles for robust normalization (5-95% range)
            low_percentile = torch.quantile(single_map.flatten(), 0.05)
            high_percentile = torch.quantile(single_map.flatten(), 0.95)
            
            # Stretch the map based on percentiles
            stretched_map = (single_map - low_percentile) / (high_percentile - low_percentile + 1e-8)
            stretched_map = torch.clamp(stretched_map, 0.05, 0.95)  # Avoid extreme values
            
            # Apply gentle gamma correction for better detail distribution
            mid_level = torch.mean(stretched_map)
            gamma = 0.9 if mid_level < 0.5 else 1.1  # Subtle gamma correction
            
            corrected_map = torch.pow(stretched_map, gamma)
            batch_maps.append(corrected_map.unsqueeze(0).unsqueeze(0))
        
        # Combine batch results
        normalized_maps = torch.cat(batch_maps, dim=0)
        
        # Apply local contrast enhancement with smaller effect
        kernel_size = 5
        local_mean = F.avg_pool2d(normalized_maps, kernel_size, stride=1, padding=kernel_size//2)
        enhanced_maps = normalized_maps + 0.3 * (normalized_maps - local_mean)
        enhanced_maps = torch.clamp(enhanced_maps, 0.1, 0.9)  # Prevent extreme values
        
        return enhanced_maps        
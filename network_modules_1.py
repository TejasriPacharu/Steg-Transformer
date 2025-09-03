import torch
import torch.nn as nn
import torch.nn.functional as F
from base_modules import RSTB, PatchEmbed, PatchUnEmbed
from utils import to_2tuple

# Enhanced Hiding Network with Dual Attention
class EnhancedHidingNetwork(nn.Module):
    """
    Hiding network that uses both cover and secret attention maps
    to guide embedding strength intelligently with improved color preservation
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
        
        # Secret information preservation module
        self.secret_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
            
        # Residual features from cover image with color-preserving architecture
        self.cover_feat_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Increased from 32 to 64
            nn.LeakyReLU(0.2)
        )

        # Secret feature extractor with focus on color preservation
        self.secret_feat_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Increased from 32 to 64
            nn.LeakyReLU(0.2)
        )
        
        # Color preservation branch for cover image
        self.color_preservation = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),  # Lightweight color extractor
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Increased from 16 to 32
            nn.LeakyReLU(0.2)
        )
        
        # Enhanced secret color preservation module
        self.secret_color_preserver = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Increased from 16 to 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh()  # Allow bidirectional adjustments
        )
        
        # Structure-preserving module for secret image
        self.structure_preserver = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # Larger kernel to capture structure
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        
        # Final processing with skip connection from cover features
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim + 64 + 64 + 32 + 16, 128, kernel_size=3, padding=1),  # Updated dimensions
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Use tanh for residual to constrain values to [-1, 1] range
        )

        # Dynamic alpha parameter for better cover-container balance
        self.alpha_base = nn.Parameter(torch.tensor(0.6))  # Decreased to allow more secret embedding
        
        # Alpha modulation based on attention
        self.alpha_modulator = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Adaptive secret strength modulator
        self.secret_strength_modulator = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Initialize minimum embedding constant - increased for better secret recovery
        self.min_embed_constant = 0.15  # Was 0.05
    
    def forward(self, cover_img, secret_img, embedding_map):
        B, C, H, W = cover_img.shape
        
        # Extract features from cover for later skip connection
        cover_features = self.cover_feat_extractor(cover_img)
        
        # Extract features from secret for enhancing detail preservation
        secret_features = self.secret_feat_extractor(secret_img)
        
        # Extract color features from cover for better color preservation
        color_features = self.color_preservation(cover_img)
        
        # Preserve color information of secret image
        secret_color_preserved = self.secret_color_preserver(secret_img)
        
        # Extract structure information from secret image
        secret_structure = self.structure_preserver(secret_img)
        
        # Create a dynamic alpha map based on embedding map
        # Areas with high embedding get lower alpha (more modification)
        alpha_map = self.alpha_modulator(embedding_map)
        dynamic_alpha = self.alpha_base * alpha_map
        
        # Encode secret information for better preservation
        secret_encoded = self.secret_encoder(secret_img)
        
        # Calculate adaptive secret strength based on secret image content
        secret_importance = self.secret_strength_modulator(secret_img)
        
        # Expand embedding map to match secret channels for element-wise multiplication
        embedding_map_expanded = embedding_map.expand(-1, C, -1, -1)
        
        # Weight secret image by embedding map with stronger effect
        weighted_secret = secret_img * (embedding_map_expanded + 0.1)  # Added 0.1 to preserve more secret information
        
        # Combine cover, weighted secret, and embedding map
        combined = torch.cat([cover_img, weighted_secret, embedding_map], dim=1)
        
        # Initial embedding
        x = self.initial_conv(combined)
        
        # Process through RSTB blocks
        for layer in self.layers:
            x = layer(x)
        
        # Combine with cover features, secret features, color features and structure features
        x = torch.cat([x, cover_features, secret_features, color_features, secret_structure], dim=1)
        
        # Final processing - output with tanh for better color dynamics
        residual = self.final_conv(x)
        
        # Dynamic blending with position-specific alpha - adjusted range for better secret preservation
        # Alpha range: 0.6 to 0.85 (lower values allow more secret information to be embedded)
        dynamic_alpha_expanded = 0.6 + 0.25 * dynamic_alpha.expand(-1, C, -1, -1)
        container = dynamic_alpha_expanded * cover_img + (1 - dynamic_alpha_expanded) * ((residual + 1) / 2)
        
        # Direct secret contribution with more aggressive scaling
        # Apply different embedding strengths for each color channel to help preserve color information
        min_embed_strength = torch.ones((B, 3, 1, 1), device=embedding_map.device) * self.min_embed_constant
        min_embed_strength[:, 0, :, :] *= 1.2  # More emphasis on red channel
        min_embed_strength[:, 1, :, :] *= 1.1  # Slight emphasis on green channel
        min_embed_strength[:, 2, :, :] *= 1.3  # Even more emphasis on blue channel to fix color issues
        
        max_embed_strength = torch.ones((B, 3, 1, 1), device=embedding_map.device) * 0.25  # Increased from 0.15
        
        # Scale the embedding map to determine secret contribution strength
        contribution_strength = min_embed_strength + (max_embed_strength - min_embed_strength) * embedding_map_expanded
        
        # Enhance contribution strength in areas important for the secret image
        contribution_strength = contribution_strength * (1.0 + 0.5 * secret_importance.expand(-1, C, -1, -1))
        
        # Apply the direct secret contribution using both color-preserved secret and encoded features
        # Use scaled sigmoid for a smoother embedding that preserves the secret
        secret_signal = torch.sigmoid(8 * (secret_color_preserved))  # Increased scaling factor for more pronounced effect
        
        # Add direct secret contribution to container
        secret_contribution = contribution_strength * secret_signal
        container = container + secret_contribution
        
        # Add additional modulated secret information using encoded features
        secret_encoded_norm = torch.tanh(secret_encoded[:, :3]) * 0.05  # Limit the effect and ensure 3 channels
        container = container + secret_encoded_norm * embedding_map_expanded * secret_importance.expand(-1, C, -1, -1)
        
        # Clamp to valid image range
        container = torch.clamp(container, 0, 1)
        
        return container


class EnhancedExtractionNetwork(nn.Module):
    def __init__(self, img_size=144, window_size=8, embed_dim=128, depths=[6, 6, 6, 6],
                 num_heads=[8, 8, 8, 8], mlp_ratio=4.):
        super().__init__()
        
        # Add a more powerful initial feature extraction with multi-scale perception
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),  # Larger kernel for broader context
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
        
        # Advanced multi-level feature extraction with skip connections
        self.skip_features = nn.ModuleList()
        for i in range(len(depths)):
            self.skip_features.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, 32, kernel_size=3, padding=1),  # Increased from 16 to 32
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(32, 32, kernel_size=1)
                )
            )
        
        # Enhanced attention mechanism with spatial and channel attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=7, padding=3),  # Larger receptive field
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Multi-scale feature enhancement
        self.multi_scale_features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=5, padding=2),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=7, padding=3),
                nn.LeakyReLU(0.2)
            )
        ])
        
        # Improved color correction module with chromatic adaptation
        self.color_corrector = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # Larger kernel for global color context
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh()  # Allow both positive and negative color adjustments
        )
        
        # Advanced color transfer module with reference-based correction
        self.color_transfer = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),  # 3 channels input + 3 channels of statistics
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh()
        )
        
        # Enhanced channel calibration module
        self.channel_calibration = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid()  # Output channel calibration weights
        )
        
        # Global color statistics modeling
        self.global_color_stats = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 12, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(12, 6, kernel_size=1),
            nn.Sigmoid()  # Output global color statistics
        )
        
        # Convert back to image space
        self.conv_out = nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1)
        
        # Multi-branch feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(64 + 32 * len(depths) + embed_dim // 2 * 3, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Progressive decoder for initial image extraction
        self.progressive_decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Multi-stage refinement modules
        self.refinement_stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(9, 32, kernel_size=3, padding=1),  # 3 (current) + 3 (container) + 3 (previous)
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(16, 3, kernel_size=3, padding=1),
                nn.Tanh()
            ) for _ in range(3)  # 3 refinement stages
        ])
        
        # Detail enhancement module
        self.detail_enhancer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Color balance correction
        self.color_balance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 12, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(12, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final color normalization layer
        self.color_normalizer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, container_img):
        # Initial feature extraction
        x = self.initial_conv(container_img)
        
        # Process through RSTB blocks with enhanced skip connections
        skip_outputs = []
        for i, layer in enumerate(self.layers):
            x_prev = x  # Save the input to this layer
            x = layer(x)
            skip_outputs.append(self.skip_features[i](x_prev))
        
        # Apply dual attention mechanism
        spatial_attn = self.spatial_attention(x)
        channel_attn = self.channel_attention(x)
        
        # Apply both attention types
        x = x * spatial_attn * channel_attn + x  # Residual connection
        
        # Extract multi-scale features
        multi_scale_feats = [module(x) for module in self.multi_scale_features]
        
        # Convert to image space
        x_out = self.conv_out(x)
        
        # Concatenate skip features and multi-scale features
        for skip in skip_outputs:
            x_out = torch.cat([x_out, skip], dim=1)
        
        for feat in multi_scale_feats:
            x_out = torch.cat([x_out, feat], dim=1)
        
        # Fuse all features
        x_fused = self.feature_fusion(x_out)
        
        # Initial secret extraction
        initial_secret = self.progressive_decoder(x_fused)
        
        # Multi-stage progressive refinement
        current_secret = initial_secret
        
        # Obtain global color statistics
        global_stats = self.global_color_stats(current_secret)
        global_stats_expanded = global_stats.expand(-1, -1, current_secret.size(2), current_secret.size(3))
        
        # Apply color transfer with statistics
        transfer_input = torch.cat([current_secret, global_stats_expanded[:, :3, :, :]], dim=1)
        color_transferred = self.color_transfer(transfer_input)
        current_secret = current_secret + 0.2 * color_transferred
        
        # Apply progressive refinement stages
        for i, refine_module in enumerate(self.refinement_stages):
            # Create input for refinement (current + container + previous if available)
            if i == 0:
                refine_input = torch.cat([current_secret, container_img, initial_secret], dim=1)
            else:
                refine_input = torch.cat([current_secret, container_img, initial_secret], dim=1)
            
            # Apply refinement
            refinement = refine_module(refine_input)
            
            # Add refinement with decreasing strength at each stage
            weight = 0.2 / (i + 1)  # 0.2, 0.1, 0.067
            current_secret = torch.clamp(current_secret + weight * refinement, 0, 1)
            
            # Apply color calibration at each stage
            calibration_weights = self.channel_calibration(current_secret)
            current_secret = current_secret * (0.9 + 0.2 * calibration_weights)  # Range: 0.9 to 1.1
        
        # Apply detail enhancement for final sharpness
        detail_enhancement = self.detail_enhancer(current_secret)
        enhanced_secret = current_secret + 0.1 * detail_enhancement
        
        # Apply color balance correction for final color adjustment
        color_multipliers = self.color_balance(enhanced_secret)
        balanced_secret = enhanced_secret * (0.9 + 0.2 * color_multipliers)  # Range: 0.9 to 1.1
        
        # Apply color correction for better RGB distribution
        color_correction = self.color_corrector(balanced_secret)
        color_corrected = balanced_secret + 0.15 * color_correction
        
        # Final color normalization
        final_secret = self.color_normalizer(color_corrected)
        
        # Ensure final output is properly bounded
        final_secret = torch.clamp(final_secret, 0, 1)
        
        return final_secret
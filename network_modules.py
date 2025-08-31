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
            
        # Residual features from cover image with color-preserving architecture
        self.cover_feat_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )

        # Secret feature extractor with focus on color preservation
        self.secret_feat_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Color preservation branch for cover image
        self.color_preservation = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),  # Lightweight color extractor
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Final processing with skip connection from cover features
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim + 32 + 32 + 16, 96, kernel_size=3, padding=1),  # Added color preservation
            nn.LeakyReLU(0.2),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Use tanh for residual to constrain values to [-1, 1] range
        )

        # Dynamic alpha parameter for better cover-container balance
        self.alpha_base = nn.Parameter(torch.tensor(0.7))  # Increased for better cover preservation
        
        # Alpha modulation based on attention
        self.alpha_modulator = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, cover_img, secret_img, embedding_map):
        B, C, H, W = cover_img.shape
        
        # Extract features from cover for later skip connection
        cover_features = self.cover_feat_extractor(cover_img)
        
        # Extract features from secret for enhancing detail preservation
        secret_features = self.secret_feat_extractor(secret_img)
        
        # Extract color features from cover for better color preservation
        color_features = self.color_preservation(cover_img)
        
        # Create a dynamic alpha map based on embedding map
        # Areas with high embedding get lower alpha (more modification)
        alpha_map = self.alpha_modulator(embedding_map)
        dynamic_alpha = self.alpha_base * alpha_map
        
        # Expand embedding map to match secret channels for element-wise multiplication
        embedding_map_expanded = embedding_map.expand(-1, C, -1, -1)
        
        # Weight secret image by embedding map with gradual effect
        weighted_secret = secret_img * embedding_map_expanded
        
        # Combine cover, weighted secret, and embedding map
        combined = torch.cat([cover_img, weighted_secret, embedding_map], dim=1)
        
        # Initial embedding
        x = self.initial_conv(combined)
        
        # Process through RSTB blocks
        for layer in self.layers:
            x = layer(x)
        
        # Combine with cover features, secret features and color features
        x = torch.cat([x, cover_features, secret_features, color_features], dim=1)
        
        # Final processing - output with tanh for better color dynamics
        residual = self.final_conv(x)
        
        # Dynamic blending with position-specific alpha - increased alpha range for better cover preservation
        # Alpha range: 0.7 to 0.9 (higher values preserve more of the cover)
        dynamic_alpha_expanded = 0.7 + 0.2 * dynamic_alpha.expand(-1, C, -1, -1)
        container = dynamic_alpha_expanded * cover_img + (1 - dynamic_alpha_expanded) * ((residual + 1) / 2)
        
        # Direct secret contribution with more careful scaling
        # Updated embedding strength to align with EnhancedSteganographySystem values
        min_embed_strength = 0.05  # Increased for better secret preservation
        max_embed_strength = 0.15  # Increased for better secret preservation
        
        # Scale the embedding map to determine secret contribution strength
        contribution_strength = min_embed_strength + (max_embed_strength - min_embed_strength) * embedding_map_expanded
        
        # Apply the direct secret contribution with the secret features to guide embedding
        # Use scaled sigmoid for a smoother embedding that preserves the secret
        secret_signal = torch.sigmoid(5 * (secret_img - 0.5))  # Sharper sigmoid for better contrast
        secret_contribution = contribution_strength * secret_signal
        
        # Add contribution and clamp to valid image range
        container = torch.clamp(container + secret_contribution, 0, 1)
        
        return container

class EnhancedExtractionNetwork(nn.Module):
    def __init__(self, img_size=144, window_size=8, embed_dim=128, depths=[6, 6, 6, 6],
                 num_heads=[8, 8, 8, 8], mlp_ratio=4.):
        super().__init__()
        
        # Add a more powerful initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Extra conv layer
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
        
        # Add direct skip connections from intermediate layers
        self.skip_features = nn.ModuleList()
        for i in range(len(depths)):
            self.skip_features.append(
                nn.Conv2d(embed_dim, 16, kernel_size=1)  # Projection for skip features
            )
        
        # Add attention mechanism to focus on areas with embedded data
        self.attention_gate = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Color correction module to address color distortion
        self.color_corrector = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),  # Large kernel for global color context
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Tanh()  # Allow both positive and negative color adjustments
        )
        
        # Convert back to image space
        self.conv_out = nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1)
        
        # Final processing with concatenated skip features
        self.final_conv = nn.Sequential(
            nn.Conv2d(64 + 16 * len(depths), 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Added sigmoid for proper 0-1 range
        )
        
        # Add residual refinement layer for final detail enhancement
        self.refine = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),  # Input: concat of initial extraction and container
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Tanh to allow positive and negative refinements
        )
        
        # Contrast enhancement module
        self.contrast_enhancer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Add color balance correction module
        self.color_balance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling for color statistics
            nn.Conv2d(3, 8, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 3, kernel_size=1),
            nn.Sigmoid()  # Output color multipliers
        )
    
    def forward(self, container_img):
        # Initial feature extraction
        x = self.initial_conv(container_img)
        
        # Process through RSTB blocks with skip connections
        skip_outputs = []
        for i, layer in enumerate(self.layers):
            x_prev = x  # Save the input to this layer
            x = layer(x)
            skip_outputs.append(self.skip_features[i](x_prev))
        
        # Generate attention map for focusing on embedded data
        attention_map = self.attention_gate(x)
        x = x * attention_map + x  # Apply soft attention (residual)
        
        # Convert to image space
        x = self.conv_out(x)
        
        # Concatenate skip features
        for skip in skip_outputs:
            x = torch.cat([x, skip], dim=1)
            
        # First-pass extraction (with sigmoid)
        initial_secret = self.final_conv(x)
        
        # Refinement using both the container and initial extraction
        refine_input = torch.cat([initial_secret, container_img], dim=1)
        refinement = self.refine(refine_input)
        
        # Apply refinement with increased strength for better detail recovery
        secret_with_refinement = initial_secret + 0.2 * refinement  # Reduced from 0.3
        
        # Apply color correction
        color_adjustment = self.color_corrector(secret_with_refinement)
        color_corrected = secret_with_refinement + 0.15 * color_adjustment  # Reduced from 0.2
        
        # Apply contrast enhancement
        contrast_adjustment = self.contrast_enhancer(color_corrected)
        enhanced_secret = color_corrected + 0.1 * contrast_adjustment  # Reduced from 0.15
        
        # Apply color balance correction
        color_multipliers = self.color_balance(enhanced_secret)
        balanced_secret = enhanced_secret * (0.8 + 0.4 * color_multipliers)  # Range: 0.8 to 1.2
        
        # Normalize histogram to improve contrast
        b, c, h, w = balanced_secret.shape
        normalized_secret = balanced_secret.clone()
        
        for i in range(b):
            for j in range(c):
                # Get channel
                channel = balanced_secret[i, j]
                
                # Get percentiles for robust normalization
                low_val = torch.quantile(channel.flatten(), 0.01)  # Reduced from 0.02
                high_val = torch.quantile(channel.flatten(), 0.99)  # Increased from 0.98
                
                # Apply normalization with reduced stretching
                normalized = 0.05 + 0.9 * (channel - low_val) / (high_val - low_val + 1e-6)  # Range: 0.05 to 0.95
                normalized_secret[i, j] = torch.clamp(normalized, 0, 1)
        
        # Ensure final output is properly bounded
        final_secret = torch.clamp(normalized_secret, 0, 1)
        
        return final_secret
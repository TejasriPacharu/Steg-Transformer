import torch
import torch.nn as nn
import torch.nn.functional as F
from base_modules import RSTB, PatchEmbed, PatchUnEmbed, BasicLayer
from utils import to_2tuple


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
            nn.Conv2d(7, 64, kernel_size=3, padding=1),p
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
    def __init__(self, img_size=144, embed_dim=128, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        
        # Initial feature extraction
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Color-preserving branch (strengthened)
        self.color_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Transformer blocks (mirror of HidingNet)
        self.transformer_blocks = nn.ModuleList()
        for depth, num_head in zip(depths, num_heads):
            block = BasicLayer(
                dim=embed_dim,
                input_resolution=(img_size, img_size),
                depth=depth,
                num_heads=num_head,
                window_size=window_size
            )
            self.transformer_blocks.append(block)
        
        # Residual refinement
        self.refinement_blocks = nn.ModuleList([
            ResidualBlock(embed_dim),
            ResidualBlock(embed_dim),
            ResidualBlock(embed_dim)
        ])
        
        # Fusion attention
        self.fusion_attention = FusionAttention(embed_dim + 64)
        
        # Final reconstruction
        self.reconstruct = nn.Sequential(
            nn.Conv2d(embed_dim + 64, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim // 4, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Optional auxiliary heads for multi-scale supervision
        self.aux_head1 = nn.Conv2d(embed_dim, 3, kernel_size=3, padding=1)
        self.aux_head2 = nn.Conv2d(96, 3, kernel_size=3, padding=1)

    def forward(self, x, attention_map=None, return_aux=False):
        # Color-preserving features
        color_features = self.color_branch(x)
        
        # Main transformer features
        features = self.init_conv(x)
        B, C, H, W = features.shape
        features_flat = features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        for block in self.transformer_blocks:
            features_flat = block(features_flat)
        
        features = features_flat.transpose(1, 2).reshape(B, C, H, W)
        
        # Refinement
        for block in self.refinement_blocks:
            features = block(features)
        
        # Auxiliary predictions (multi-scale supervision)
        aux1 = torch.sigmoid(self.aux_head1(features))
        
        # Merge with color branch
        merged_features = torch.cat([features, color_features], dim=1)
        merged_features = self.fusion_attention(merged_features)
        
        aux2 = torch.sigmoid(self.aux_head2(merged_features))
        
        # Final reconstruction
        output = self.reconstruct(merged_features)
        
        # Attention map refinement (optional)
        if attention_map is not None:
            output = output * (0.5 + 0.5 * attention_map.expand_as(output))
        
        if return_aux:
            return output, aux1, aux2
        else:
            return output
            
class FusionAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        ca = self.channel_att(x)
        sa = self.spatial_att(x)
        return x * ca * sa


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

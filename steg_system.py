import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from attention_modules import DualAttentionHeatmapGenerator, AttentionHeatmapGenerator
from network_modules import EnhancedHidingNetwork, EnhancedExtractionNetwork
from utils import calculate_psnr, calculate_ssim, to_2tuple

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
        max_strength = 0.9  # Maximum embedding strength
        
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
    
    def forward_hide(self, cover_img, secret_img, use_high_attention=True, return_maps=False):
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

    # Add this to the EnhancedSteganographySystem class
    def attention_loss(self, cover_attention, secret_attention):
        # Encourage diversity in attention maps
        # Penalize uniform attention (too high or too low values across the map)
        cover_mean = cover_attention.mean()
        cover_std = cover_attention.std()
        secret_mean = secret_attention.mean()
        secret_std = secret_attention.std()
        
        # Encourage attention maps with good standard deviation (not uniform)
        diversity_loss = torch.exp(-cover_std*5) + torch.exp(-secret_std*5)
        return diversity_loss
        
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

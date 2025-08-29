import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from attention_modules import DualAttentionHeatmapGenerator
from network_modules import EnhancedHidingNetwork, EnhancedExtractionNetwork
from utils import calculate_psnr, calculate_ssim, to_2tuple

# Complete Enhanced Steganography System
class EnhancedSteganographySystem(nn.Module):
    """
    Complete end-to-end steganography system that compares embedding in
    high vs low attention areas and evaluates results with improved
    detail preservation and attention diversity
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
        
        # Detail-aware embedding strength modulator
        self.embed_modulator = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def compute_embedding_map(self, cover_attention, secret_attention, use_high_attention=True):
        """
        Compute optimal embedding map based on both attention maps with improved
        detail preservation and gradient sensitivity
        """
        B, _, H, W = cover_attention.shape
        
        # Calculate gradient maps for cover and secret to identify detail areas
        cover_grad_x = F.conv2d(
            F.pad(cover_attention, (1, 1, 0, 0), mode='replicate'),
            torch.tensor([[-1, 0, 1]], dtype=torch.float32).view(1, 1, 1, 3).to(cover_attention.device),
            padding=0
        )
        
        cover_grad_y = F.conv2d(
            F.pad(cover_attention, (0, 0, 1, 1), mode='replicate'),
            torch.tensor([[-1], [0], [1]], dtype=torch.float32).view(1, 1, 3, 1).to(cover_attention.device),
            padding=0
        )
        
        # Calculate gradient magnitude
        cover_gradient = torch.sqrt(cover_grad_x.pow(2) + cover_grad_y.pow(2) + 1e-6)
        
        # Get high detail regions in the secret image (high gradient areas)
        secret_grad_x = F.conv2d(
            F.pad(secret_attention, (1, 1, 0, 0), mode='replicate'),
            torch.tensor([[-1, 0, 1]], dtype=torch.float32).view(1, 1, 1, 3).to(secret_attention.device),
            padding=0
        )
        
        secret_grad_y = F.conv2d(
            F.pad(secret_attention, (0, 0, 1, 1), mode='replicate'),
            torch.tensor([[-1], [0], [1]], dtype=torch.float32).view(1, 1, 3, 1).to(secret_attention.device),
            padding=0
        )
        
        secret_gradient = torch.sqrt(secret_grad_x.pow(2) + secret_grad_y.pow(2) + 1e-6)
        
        # Normalize the gradients
        cover_gradient = (cover_gradient - cover_gradient.min()) / (cover_gradient.max() - cover_gradient.min() + 1e-8)
        secret_gradient = (secret_gradient - secret_gradient.min()) / (secret_gradient.max() - secret_gradient.min() + 1e-8)
        
        if use_high_attention:
            # Embed more in high attention areas but less in high gradient areas of cover
            # to preserve cover details, while focusing on high detail areas in secret
            embedding_strength = (cover_attention - 0.3 * cover_gradient) * (secret_attention + 0.4 * secret_gradient + 0.25)
        else:
            # Embed in low attention areas but still consider detail preservation
            # Avoid very low attention areas in secret (add 0.25 baseline)
            embedding_strength = (1 - cover_attention + 0.2 * cover_gradient) * (secret_attention + 0.4 * secret_gradient + 0.25)
            
        # Apply spatial consistency to avoid abrupt changes in embedding strength
        kernel_size = 3
        padding = kernel_size // 2
        smoothed_strength = F.avg_pool2d(embedding_strength, kernel_size, stride=1, padding=padding)
        embedding_strength = 0.7 * embedding_strength + 0.3 * smoothed_strength
        
        # Apply adaptive normalization for better balance
        embedding_strength = self.normalize_embedding_strength(embedding_strength)
        
        return embedding_strength
        
    def normalize_embedding_strength(self, strength_map):
        """
        Adaptive scaling to ensure enough embedding capacity
        while maintaining higher visual quality with improved
        dynamic range compression
        """
        B, _, H, W = strength_map.shape
        
        # Increase minimum strength to ensure adequate information preservation
        min_strength = 0.5  # Base minimum strength
        max_strength = 0.95  # Maximum strength cap
        
        # Apply histogram equalization-like processing to each map individually
        normalized_maps = []
        for b in range(B):
            # Extract single map
            single_map = strength_map[b, 0]
            
            # Calculate percentiles for robust normalization
            low_percentile = torch.quantile(single_map.flatten(), 0.05)  # 5th percentile
            high_percentile = torch.quantile(single_map.flatten(), 0.95)  # 95th percentile
            
            # Stretch the map using percentiles to avoid outlier influence
            stretched = (single_map - low_percentile) / (high_percentile - low_percentile + 1e-8)
            stretched = torch.clamp(stretched, 0, 1)
            
            # Apply smooth sigmoid-like function to enhance mid-tones while preserving details
            # Using a custom function that maintains more gradations than pure sigmoid
            enhanced = (torch.sin((stretched - 0.5) * np.pi) + 1) / 2
            
            # Scale to desired strength range
            scaled = min_strength + (max_strength - min_strength) * enhanced
            
            # Add small constant to ensure minimum embedding everywhere
            scaled = scaled + 0.05
            scaled = torch.clamp(scaled, 0, max_strength)
            
            normalized_maps.append(scaled.unsqueeze(0).unsqueeze(0))
        
        return torch.cat(normalized_maps, dim=0)
    
    def forward_hide(self, cover_img, secret_img, use_high_attention=True, return_maps=False):
        """
        Forward pass for hiding process with improved attention diversity
        """
        # Generate attention maps for both images
        cover_attention = self.cover_attention(cover_img)
        secret_attention = self.secret_attention(secret_img)
        
        # Ensure attention maps have detail and diversity
        cover_attention = self.enhance_attention_diversity(cover_attention)
        secret_attention = self.enhance_attention_diversity(secret_attention)
        
        # Compute embedding map
        embedding_map = self.compute_embedding_map(
            cover_attention, secret_attention, use_high_attention
        )
        
        # Generate container image
        container = self.hiding_network(cover_img, secret_img, embedding_map)
    
        return container, cover_attention, secret_attention, embedding_map
    
    def enhance_attention_diversity(self, attention_map):
        """
        Enhance attention map diversity to avoid uniform areas
        """
        B, C, H, W = attention_map.shape
        
        # Calculate batch statistics
        flat_attn = attention_map.view(B, -1)
        means = flat_attn.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        stds = flat_attn.std(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        
        # For very uniform maps (low std), apply contrast enhancement
        # This prevents completely white or black attention maps
        enhanced_maps = []
        for b in range(B):
            map_b = attention_map[b]
            std_b = stds[b]
            
            if std_b < 0.1:  # If map is too uniform
                # Apply histogram equalization-like enhancement
                sorted_vals, _ = torch.sort(map_b.flatten())
                n = sorted_vals.shape[0]
                ranks = torch.linspace(0, 1, n, device=map_b.device)
                
                # Map original values to rank-based values (0-1 range)
                # This is similar to histogram equalization
                map_b_flat = map_b.flatten()
                enhanced_flat = torch.zeros_like(map_b_flat)
                
                for i, val in enumerate(sorted_vals):
                    mask = (map_b_flat == val)
                    enhanced_flat[mask] = ranks[i]
                
                # Reshape back and add to result
                enhanced_maps.append(enhanced_flat.view(C, H, W))
            else:
                enhanced_maps.append(map_b)
        
        return torch.stack(enhanced_maps)
    
    def forward_extract(self, container):
        """
        Forward pass for extraction process
        """
        return self.extraction_network(container)

    def attention_loss(self, cover_attention, secret_attention):
        """
        Encourage diversity in attention maps and avoid uniform attention
        """
        # Compute statistics of attention maps
        cover_mean = cover_attention.mean(dim=[2, 3])  # Mean across spatial dimensions
        cover_std = cover_attention.std(dim=[2, 3])    # Std across spatial dimensions
        secret_mean = secret_attention.mean(dim=[2, 3])
        secret_std = secret_attention.std(dim=[2, 3])
        
        # Calculate entropy of attention maps for diversity measure
        # Approximate entropy using binning
        def approximate_entropy(attention):
            bins = 10
            batch_entropy = []
            
            for b in range(attention.shape[0]):
                # Flatten spatial dimensions
                flat_attn = attention[b].view(-1)
                
                # Create histogram
                hist = torch.histc(flat_attn, bins=bins, min=0, max=1)
                hist = hist / hist.sum()  # Normalize
                
                # Calculate entropy (-sum(p*log(p)))
                # Avoid log(0) issues
                entropy = -torch.sum(hist * torch.log2(hist + 1e-10))
                batch_entropy.append(entropy)
                
            return torch.stack(batch_entropy)
        
        cover_entropy = approximate_entropy(cover_attention)
        secret_entropy = approximate_entropy(secret_attention)
        
        # Encourage high standard deviation (diverse attention maps)
        std_loss = torch.exp(-5 * cover_std) + torch.exp(-5 * secret_std)
        
        # Encourage high entropy (more information in attention maps)
        max_entropy = torch.log2(torch.tensor(10.0, device=cover_attention.device))  # Max possible entropy
        entropy_loss = torch.exp(-3 * (cover_entropy / max_entropy)) + torch.exp(-3 * (secret_entropy / max_entropy))
        
        # Combine losses
        diversity_loss = std_loss.mean() + entropy_loss.mean()
        
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

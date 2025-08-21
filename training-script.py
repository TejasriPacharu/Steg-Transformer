import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
from datetime import datetime
import torchvision
import torch.nn.functional as F

# Import the enhanced models from swin_model.py
from enhanced_swin_model import (
    DualAttentionHeatmapGenerator, 
    EnhancedHidingNetwork, 
    EnhancedExtractionNetwork, 
    EnhancedSteganographySystem,
    calculate_psnr, 
    calculate_ssim, 
    calculate_mse
)

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with tiny-imagenet-200
            split (string): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
   
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.img_paths = []

        if not os.path.exists(self.root_dir):
            print(f"{split.capitalize()} directory '{self.root_dir}' does not exist.")
            return

        for img_file in os.listdir(self.root_dir):
            if img_file.endswith('.JPEG'):
                self.img_paths.append(os.path.join(self.root_dir, img_file))

        print(f"Loaded {len(self.img_paths)} images for {split}") 

    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        try:
            # Load the cover image
            cover_image = Image.open(img_path).convert('RGB')
            
            # For the secret image, select a different random image
            secret_idx = random.randint(0, len(self.img_paths) - 1)
            # Make sure it's not the same as the cover image
            while secret_idx == idx:
                secret_idx = random.randint(0, len(self.img_paths) - 1)
                
            secret_img_path = self.img_paths[secret_idx]
            secret_image = Image.open(secret_img_path).convert('RGB')
            
            if self.transform:
                cover_image = self.transform(cover_image)
                secret_image = self.transform(secret_image)
            
            # Return different images for cover and secret
            return cover_image, secret_image
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return default black images in case of error
            if self.transform:
                return self.transform(Image.new('RGB', (64, 64), (0, 0, 0))), self.transform(Image.new('RGB', (64, 64), (0, 0, 0)))
            else:
                return Image.new('RGB', (64, 64), (0, 0, 0)), Image.new('RGB', (64, 64), (0, 0, 0))

class CombinedLoss(nn.Module):
    """
    Combined loss function that includes MSE, perceptual loss, and
    edge-preservation components for better image quality and detail preservation.
    """
    def __init__(self, alpha=0.7):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
        # Initialize a small VGG-like network for perceptual loss
        # We use only initial layers to focus on low-level features
        self.perceptual_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Edge detection kernels (Sobel)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    
    def forward(self, predicted, target):
        # Basic MSE loss
        mse_loss = self.mse(predicted, target)
        
        # Perceptual loss (feature-based)
        with torch.no_grad():
            target_features = self.perceptual_layers(target)
        predicted_features = self.perceptual_layers(predicted)
        perceptual_loss = self.mse(predicted_features, target_features)
        
        # Edge preservation loss
        if self.sobel_x.device != predicted.device:
            self.sobel_x = self.sobel_x.to(predicted.device)
            self.sobel_y = self.sobel_y.to(predicted.device)
        
        # Calculate edges using Sobel
        target_edges_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_edges_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        target_edges = torch.sqrt(target_edges_x.pow(2) + target_edges_y.pow(2) + 1e-6)
        
        pred_edges_x = F.conv2d(predicted, self.sobel_x, padding=1, groups=3)
        pred_edges_y = F.conv2d(predicted, self.sobel_y, padding=1, groups=3)
        pred_edges = torch.sqrt(pred_edges_x.pow(2) + pred_edges_y.pow(2) + 1e-6)
        
        edge_loss = F.l1_loss(pred_edges, target_edges)
        
        # Combine losses with weighting
        combined_loss = mse_loss + self.alpha * perceptual_loss + (1 - self.alpha) * edge_loss
        
        return combined_loss


def calculate_psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)
        
    Returns:
        PSNR value in dB
    """
    # Ensure images are in the same range (0-1)
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)
        
    Returns:
        SSIM value (between -1 and 1, higher is better)
    """
    # Ensure images are in the same range (0-1)
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    # Constants for stability
    C1 = (0.01 * 1.0)**2
    C2 = (0.03 * 1.0)**2
    
    # Calculate means
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Calculate variances and covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
    
    # Calculate SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim = numerator / denominator
    
    return ssim


def validate(steg_system, dataloader, device, alpha=0.5, beta=0.5, use_high_attention=True):
    """
    Validate the model using the enhanced steganography system
    
    Args:
        steg_system: EnhancedSteganographySystem model
        dataloader: DataLoader for the validation set
        device: Device to use (cuda/cpu)
        alpha: Weight for cover-container loss
        beta: Weight for secret-extracted loss
        use_high_attention: Whether to use high attention regions (True) or low attention regions (False)
    
    Returns:
        Validation metrics
    """
    steg_system.eval()
    
    total_loss = 0.0
    total_hiding_loss = 0.0
    total_extraction_loss = 0.0
    total_psnr_container = 0.0
    total_psnr_secret = 0.0
    total_ssim_container = 0.0
    total_ssim_secret = 0.0
    
    criterion = CombinedLoss(alpha=0.7)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validating')
        
        for i, (cover_imgs, secret_imgs) in enumerate(progress_bar):
            cover_imgs = cover_imgs.to(device)
            secret_imgs = secret_imgs.to(device)
            
            # Forward pass using the enhanced system
            container_imgs, cover_atn, secret_atn, embed_map = steg_system.forward_hide(
                cover_imgs, secret_imgs, use_high_attention=use_high_attention
            )
            extracted_secrets = steg_system.forward_extract(container_imgs)
            
            # Calculate losses
            hiding_loss = criterion(container_imgs, cover_imgs)
            extraction_loss = criterion(extracted_secrets, secret_imgs)
            
            # Combined loss
            loss = alpha * hiding_loss + beta * extraction_loss
            
            # Calculate metrics
            for j in range(cover_imgs.size(0)):
                # Convert to numpy for PSNR and SSIM calculation
                cover_np = cover_imgs[j].cpu().numpy().transpose(1, 2, 0)
                container_np = container_imgs[j].cpu().numpy().transpose(1, 2, 0)
                secret_np = secret_imgs[j].cpu().numpy().transpose(1, 2, 0)
                extracted_np = extracted_secrets[j].cpu().numpy().transpose(1, 2, 0)
                
                # Calculate PSNR (normalized [0,1] images)
                psnr_container = calculate_psnr(cover_np, container_np)
                psnr_secret = calculate_psnr(secret_np, extracted_np)
                
                # Calculate SSIM (normalized [0,1] images)
                ssim_container = calculate_ssim(cover_np, container_np)
                ssim_secret = calculate_ssim(secret_np, extracted_np)
                
                total_psnr_container += psnr_container
                total_psnr_secret += psnr_secret
                total_ssim_container += ssim_container
                total_ssim_secret += ssim_secret
            
            # Update stats
            total_loss += loss.item()
            total_hiding_loss += hiding_loss.item()
            total_extraction_loss += extraction_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr_c': f'{psnr_container:.2f}',
                'psnr_s': f'{psnr_secret:.2f}'
            })
    
    # Calculate averages
    n_samples = len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    avg_hiding_loss = total_hiding_loss / len(dataloader)
    avg_extraction_loss = total_extraction_loss / len(dataloader)
    avg_psnr_container = total_psnr_container / n_samples
    avg_psnr_secret = total_psnr_secret / n_samples
    avg_ssim_container = total_ssim_container / n_samples
    avg_ssim_secret = total_ssim_secret / n_samples
    
    return (avg_loss, avg_hiding_loss, avg_extraction_loss, 
            avg_psnr_container, avg_psnr_secret, 
            avg_ssim_container, avg_ssim_secret)

def pretrain_extraction(steg_system, train_loader, val_loader, device, num_epochs=5, lr=0.0002):
    """
    Pretrain the extraction network alone to improve its base capabilities
    before joint training with the hiding network.
    
    Args:
        steg_system: The complete steganography system
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to run training on
        num_epochs: Number of pretraining epochs
        lr: Learning rate for pretraining
    """
    print("Starting extraction network pretraining...")
    
    # We'll only train the extraction network
    # First, set all networks to eval mode
    steg_system.eval()
    
    # Then set extraction network to train mode
    steg_system.extraction_network.train()
    
    # Define optimizer just for extraction network parameters
    optimizer = optim.AdamW(steg_system.extraction_network.parameters(), lr=lr)
    
    # Define criterion (combined loss)
    criterion = CombinedLoss(alpha=1.0)  # Full weight on reconstruction quality
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        steg_system.extraction_network.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Pretraining Epoch {epoch+1}/{num_epochs}')
        
        for i, (_, secret_imgs) in enumerate(progress_bar):
            # We only need the secret images for pretraining the extractor
            secret_imgs = secret_imgs.to(device)
            
            optimizer.zero_grad()
            
            # Generate noisy container images (simulating the output of hiding network)
            # This helps the extraction network learn to be robust to various distortions
            with torch.no_grad():
                # First get container images with the current hiding network
                container_imgs, _, _, _ = steg_system.forward_hide(
                    secret_imgs, secret_imgs, use_high_attention=True
                )
                
                # Add some noise to create more challenging examples
                noise_level = 0.05 * (1.0 - epoch / num_epochs)  # Reduce noise over time
                noise = torch.randn_like(container_imgs) * noise_level
                noisy_containers = torch.clamp(container_imgs + noise, 0, 1)
            
            # Extract the secrets from the noisy containers
            extracted_secrets = steg_system.extraction_network(noisy_containers)
            
            # Calculate loss
            loss = criterion(extracted_secrets, secret_imgs)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Update stats
            train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        steg_system.extraction_network.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        
        with torch.no_grad():
            for i, (_, secret_imgs) in enumerate(val_loader):
                secret_imgs = secret_imgs.to(device)
                
                # Create containers with hiding network
                container_imgs, _, _, _ = steg_system.forward_hide(
                    secret_imgs, secret_imgs, use_high_attention=True
                )
                
                # Extract the secrets
                extracted_secrets = steg_system.extraction_network(container_imgs)
                
                # Calculate metrics
                loss = criterion(extracted_secrets, secret_imgs)
                val_loss += loss.item()
                
                # Calculate PSNR and SSIM for batch
                for j in range(secret_imgs.size(0)):
                    original = secret_imgs[j].cpu().numpy().transpose(1, 2, 0)
                    extracted = extracted_secrets[j].cpu().numpy().transpose(1, 2, 0)
                    
                    val_psnr += calculate_psnr(original, extracted)
                    val_ssim += calculate_ssim(original, extracted)
        
        # Calculate averages
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / (len(val_loader) * val_loader.batch_size)
        avg_val_ssim = val_ssim / (len(val_loader) * val_loader.batch_size)
        
        # Print results
        print(f"Pretraining Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val PSNR: {avg_val_psnr:.2f}dB")
        print(f"  Val SSIM: {avg_val_ssim:.4f}")
        
        # Save if this is the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  New best validation loss: {best_val_loss:.4f}")
            
            # Save the pretrained extraction network
            torch.save({
                'epoch': epoch + 1,
                'extraction_network_state_dict': steg_system.extraction_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_psnr': avg_val_psnr,
                'val_ssim': avg_val_ssim,
            }, 'pretrained_extraction_network.pth')
    
    print("Pretraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load the best model from pretraining
    pretrained_checkpoint = torch.load('pretrained_extraction_network.pth')
    steg_system.extraction_network.load_state_dict(pretrained_checkpoint['extraction_network_state_dict'])
    
    # Return the model to its normal state (all components trainable)
    steg_system.train()


def visualize_results(steg_system, dataloader, device, save_dir, epoch, 
                     use_high_attention=True, num_samples=5):
    """
    Visualize the results of the steganography system
    """
    steg_system.eval()
    
    # Get a batch of data
    cover_imgs, secret_imgs = next(iter(dataloader))
    cover_imgs = cover_imgs.to(device)
    secret_imgs = secret_imgs.to(device)
    
    # Limit to num_samples
    cover_imgs = cover_imgs[:num_samples]
    secret_imgs = secret_imgs[:num_samples]
    
    with torch.no_grad():
        # Get container images and attention maps
        container_imgs, cover_attention, secret_attention, embedding_map = steg_system.forward_hide(
            cover_imgs, secret_imgs, use_high_attention=use_high_attention, return_maps=True
        )
        
        # Extract secrets
        extracted_secrets = steg_system.forward_extract(container_imgs)
        
        # Create visualization grid
        fig, axes = plt.subplots(num_samples, 6, figsize=(20, 4 * num_samples))
        
        # Handle case where there's only one sample
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        attention_mode = "High" if use_high_attention else "Low"
        
        for i in range(num_samples):
            # Convert tensors to numpy for visualization
            cover_np = cover_imgs[i].cpu().numpy().transpose(1, 2, 0)
            secret_np = secret_imgs[i].cpu().numpy().transpose(1, 2, 0)
            container_np = container_imgs[i].cpu().numpy().transpose(1, 2, 0)
            extracted_np = extracted_secrets[i].cpu().numpy().transpose(1, 2, 0)
            
            # Get attention maps
            cover_attn_np = cover_attention[i, 0].cpu().numpy()
            secret_attn_np = secret_attention[i, 0].cpu().numpy()
            
            # Calculate residual (difference map x10 for visibility)
            diff = np.abs(container_np - cover_np) * 10
            
            # Calculate metrics
            psnr_container = calculate_psnr(cover_np, container_np)
            psnr_secret = calculate_psnr(secret_np, extracted_np)
            ssim_container = calculate_ssim(cover_np, container_np)
            ssim_secret = calculate_ssim(secret_np, extracted_np)
            
            # Plot images
            axes[i, 0].imshow(cover_np)
            axes[i, 0].set_title("Cover Image")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(secret_np)
            axes[i, 1].set_title("Secret Image")
            axes[i, 1].axis("off")
            
            # Plot attention maps
            axes[i, 2].imshow(cover_attn_np, cmap='hot')
            axes[i, 2].set_title(f"{attention_mode} Attention Map (Cover)")
            axes[i, 2].axis("off")
            
            axes[i, 3].imshow(container_np)
            axes[i, 3].set_title(f"Container\nPSNR: {psnr_container:.2f}dB, SSIM: {ssim_container:.4f}")
            axes[i, 3].axis("off")
            
            axes[i, 4].imshow(diff)
            axes[i, 4].set_title("Residual (x10)")
            axes[i, 4].axis("off")
            
            axes[i, 5].imshow(extracted_np)
            axes[i, 5].set_title(f"Extracted Secret\nPSNR: {psnr_secret:.2f}dB, SSIM: {ssim_secret:.4f}")
            axes[i, 5].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'epoch_{epoch+1}_{attention_mode}_attention.png'))
        plt.close()


def save_checkpoint(checkpoint_data, filepath):
    """
    Save a checkpoint to disk
    """
    torch.save(checkpoint_data, filepath)
    print(f"Checkpoint saved to {filepath}")


def train_epoch(steg_system, dataloader, optimizer, device, 
                alpha=0.3, beta=0.7, use_high_attention=True, train_both=False):
    """
    Train for one epoch using the enhanced steganography system with attention diversity loss
    """
    steg_system.train()
    
    total_loss = 0.0
    total_hiding_loss = 0.0
    total_extraction_loss = 0.0
    total_attention_loss = 0.0
    
    # For tracking metrics when training with both attention modes
    if train_both:
        opposite_total_loss = 0.0
        opposite_total_hiding_loss = 0.0
        opposite_total_extraction_loss = 0.0
        opposite_total_attention_loss = 0.0
    
    criterion = CombinedLoss(alpha=0.7)
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for i, (cover_imgs, secret_imgs) in enumerate(progress_bar):
        cover_imgs = cover_imgs.to(device)
        secret_imgs = secret_imgs.to(device)
        
        optimizer.zero_grad()
        
        # Process with primary attention mode (high or low based on use_high_attention)
        container_imgs, cover_attention, secret_attention, embedding_map = steg_system.forward_hide(
            cover_imgs, secret_imgs, use_high_attention=use_high_attention
        )
        extracted_secrets = steg_system.forward_extract(container_imgs)
        
        # Calculate standard losses for primary mode
        hiding_loss = criterion(container_imgs, cover_imgs)
        extraction_loss = criterion(extracted_secrets, secret_imgs)
        perceptual_loss = F.l1_loss(extracted_secrets, secret_imgs)
        
        # Add attention diversity loss for primary mode
        cover_std = cover_attention.view(cover_attention.size(0), -1).std(dim=1).mean()
        secret_std = secret_attention.view(secret_attention.size(0), -1).std(dim=1).mean()
        attention_loss = torch.exp(-5 * cover_std) + torch.exp(-5 * secret_std)
        
        # Combined loss for primary mode
        loss = alpha * hiding_loss + beta * (extraction_loss + 0.5 * perceptual_loss) + 0.1 * attention_loss
        
        # Process with opposite attention mode if training both
        if train_both:
            # Use the same cover and secret images but with opposite attention mode
            opposite_container_imgs, opposite_cover_attention, opposite_secret_attention, opposite_embedding_map = steg_system.forward_hide(
                cover_imgs, secret_imgs, use_high_attention=(not use_high_attention)
            )
            opposite_extracted_secrets = steg_system.forward_extract(opposite_container_imgs)
            
            # Calculate losses for opposite mode
            opposite_hiding_loss = criterion(opposite_container_imgs, cover_imgs)
            opposite_extraction_loss = criterion(opposite_extracted_secrets, secret_imgs)
            opposite_perceptual_loss = F.l1_loss(opposite_extracted_secrets, secret_imgs)
            
            # Attention diversity loss for opposite mode
            opposite_cover_std = opposite_cover_attention.view(opposite_cover_attention.size(0), -1).std(dim=1).mean()
            opposite_secret_std = opposite_secret_attention.view(opposite_secret_attention.size(0), -1).std(dim=1).mean()
            opposite_attention_loss = torch.exp(-5 * opposite_cover_std) + torch.exp(-5 * opposite_secret_std)
            
            # Combined loss for opposite mode
            opposite_loss = alpha * opposite_hiding_loss + beta * (opposite_extraction_loss + 0.5 * opposite_perceptual_loss) + 0.1 * opposite_attention_loss
            
            # Backward both losses
            loss.backward()
            opposite_loss.backward()
            
            # Update opposite mode stats
            opposite_total_loss += opposite_loss.item()
            opposite_total_hiding_loss += opposite_hiding_loss.item()
            opposite_total_extraction_loss += opposite_extraction_loss.item()
            opposite_total_attention_loss += opposite_attention_loss.item()
        else:
            # Regular backward pass for single mode
            loss.backward()
        
        optimizer.step()
        
        # Update stats for primary mode
        total_loss += loss.item()
        total_hiding_loss += hiding_loss.item()
        total_extraction_loss += extraction_loss.item()
        total_attention_loss += attention_loss.item()
        
        # Update progress bar
        if train_both:
            progress_bar.set_postfix({
                'loss': f'{(loss.item() + opposite_loss.item()) / 2:.4f}',
                'hiding': f'{(hiding_loss.item() + opposite_hiding_loss.item()) / 2:.4f}',
                'extract': f'{(extraction_loss.item() + opposite_extraction_loss.item()) / 2:.4f}'
            })
        else:
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'hiding': f'{hiding_loss.item():.4f}',
                'extract': f'{extraction_loss.item():.4f}'
            })
    
    # Calculate average losses for primary mode
    avg_loss = total_loss / len(dataloader)
    avg_hiding_loss = total_hiding_loss / len(dataloader)
    avg_extraction_loss = total_extraction_loss / len(dataloader)
    
    results = (avg_loss, avg_hiding_loss, avg_extraction_loss)
    
    # Calculate average losses for opposite mode if applicable
    if train_both:
        opposite_avg_loss = opposite_total_loss / len(dataloader)
        opposite_avg_hiding_loss = opposite_total_hiding_loss / len(dataloader)
        opposite_avg_extraction_loss = opposite_total_extraction_loss / len(dataloader)
        
        opposite_results = (opposite_avg_loss, opposite_avg_hiding_loss, opposite_avg_extraction_loss)
        return results, opposite_results
    
    return results


def validate(steg_system, dataloader, device, alpha=0.5, beta=0.5, use_high_attention=True):
    """
    Validate the model using the enhanced steganography system
    
    Args:
        steg_system: EnhancedSteganographySystem model
        dataloader: DataLoader for the validation set
        device: Device to use (cuda/cpu)
        alpha: Weight for cover-container loss
        beta: Weight for secret-extracted loss
        use_high_attention: Whether to use high attention regions (True) or low attention regions (False)
    
    Returns:
        Validation metrics
    """
    steg_system.eval()
    
    total_loss = 0.0
    total_hiding_loss = 0.0
    total_extraction_loss = 0.0
    total_psnr_container = 0.0
    total_psnr_secret = 0.0
    total_ssim_container = 0.0
    total_ssim_secret = 0.0
    
    criterion = CombinedLoss(alpha=0.7)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validating')
        
        for i, (cover_imgs, secret_imgs) in enumerate(progress_bar):
            cover_imgs = cover_imgs.to(device)
            secret_imgs = secret_imgs.to(device)
            
            # Forward pass using the enhanced system
            container_imgs, cover_atn, secret_atn, embed_map = steg_system.forward_hide(
                cover_imgs, secret_imgs, use_high_attention=use_high_attention
            )
            extracted_secrets = steg_system.forward_extract(container_imgs)
            
            # Calculate losses
            hiding_loss = criterion(container_imgs, cover_imgs)
            extraction_loss = criterion(extracted_secrets, secret_imgs)
            
            # Combined loss
            loss = alpha * hiding_loss + beta * extraction_loss
            
            # Calculate metrics
            for j in range(cover_imgs.size(0)):
                # Convert to numpy for PSNR and SSIM calculation
                cover_np = cover_imgs[j].cpu().numpy().transpose(1, 2, 0)
                container_np = container_imgs[j].cpu().numpy().transpose(1, 2, 0)
                secret_np = secret_imgs[j].cpu().numpy().transpose(1, 2, 0)
                extracted_np = extracted_secrets[j].cpu().numpy().transpose(1, 2, 0)
                
                # Calculate PSNR (normalized [0,1] images)
                psnr_container = calculate_psnr(cover_np, container_np)
                psnr_secret = calculate_psnr(secret_np, extracted_np)
                
                # Calculate SSIM (normalized [0,1] images)
                ssim_container = calculate_ssim(cover_np, container_np)
                ssim_secret = calculate_ssim(secret_np, extracted_np)
                
                total_psnr_container += psnr_container
                total_psnr_secret += psnr_secret
                total_ssim_container += ssim_container
                total_ssim_secret += ssim_secret
            
            # Update stats
            total_loss += loss.item()
            total_hiding_loss += hiding_loss.item()
            total_extraction_loss += extraction_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr_c': f'{psnr_container:.2f}',
                'psnr_s': f'{psnr_secret:.2f}'
            })
    
    # Calculate averages
    n_samples = len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    avg_hiding_loss = total_hiding_loss / len(dataloader)
    avg_extraction_loss = total_extraction_loss / len(dataloader)
    avg_psnr_container = total_psnr_container / n_samples
    avg_psnr_secret = total_psnr_secret / n_samples
    avg_ssim_container = total_ssim_container / n_samples
    avg_ssim_secret = total_ssim_secret / n_samples
    
    return (avg_loss, avg_hiding_loss, avg_extraction_loss, 
            avg_psnr_container, avg_psnr_secret, 
            avg_ssim_container, avg_ssim_secret)

def pretrain_extraction(steg_system, train_loader, val_loader, device, num_epochs=5, lr=0.0002):
    """
    Pretrain the extraction network alone to improve its base capabilities
    before joint training with the hiding network.
    
    Args:
        steg_system: The complete steganography system
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to run training on
        num_epochs: Number of pretraining epochs
        lr: Learning rate for pretraining
    """
    print("Starting extraction network pretraining...")
    
    # We'll only train the extraction network
    # First, set all networks to eval mode
    steg_system.eval()
    
    # Then set extraction network to train mode
    steg_system.extraction_network.train()
    
    # Define optimizer just for extraction network parameters
    optimizer = optim.AdamW(steg_system.extraction_network.parameters(), lr=lr)
    
    # Define criterion (combined loss)
    criterion = CombinedLoss(alpha=1.0)  # Full weight on reconstruction quality
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        steg_system.extraction_network.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Pretraining Epoch {epoch+1}/{num_epochs}')
        
        for i, (_, secret_imgs) in enumerate(progress_bar):
            # We only need the secret images for pretraining the extractor
            secret_imgs = secret_imgs.to(device)
            
            optimizer.zero_grad()
            
            # Generate noisy container images (simulating the output of hiding network)
            # This helps the extraction network learn to be robust to various distortions
            with torch.no_grad():
                # First get container images with the current hiding network
                container_imgs, _, _, _ = steg_system.forward_hide(
                    secret_imgs, secret_imgs, use_high_attention=True
                )
                
                # Add some noise to create more challenging examples
                noise_level = 0.05 * (1.0 - epoch / num_epochs)  # Reduce noise over time
                noise = torch.randn_like(container_imgs) * noise_level
                noisy_containers = torch.clamp(container_imgs + noise, 0, 1)
            
            # Extract the secrets from the noisy containers
            extracted_secrets = steg_system.extraction_network(noisy_containers)
            
            # Calculate loss
            loss = criterion(extracted_secrets, secret_imgs)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Update stats
            train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        steg_system.extraction_network.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        
        with torch.no_grad():
            for i, (_, secret_imgs) in enumerate(val_loader):
                secret_imgs = secret_imgs.to(device)
                
                # Create containers with hiding network
                container_imgs, _, _, _ = steg_system.forward_hide(
                    secret_imgs, secret_imgs, use_high_attention=True
                )
                
                # Extract the secrets
                extracted_secrets = steg_system.extraction_network(container_imgs)
                
                # Calculate metrics
                loss = criterion(extracted_secrets, secret_imgs)
                val_loss += loss.item()
                
                # Calculate PSNR and SSIM for batch
                for j in range(secret_imgs.size(0)):
                    original = secret_imgs[j].cpu().numpy().transpose(1, 2, 0)
                    extracted = extracted_secrets[j].cpu().numpy().transpose(1, 2, 0)
                    
                    val_psnr += calculate_psnr(original, extracted)
                    val_ssim += calculate_ssim(original, extracted)
        
        # Calculate averages
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / (len(val_loader) * val_loader.batch_size)
        avg_val_ssim = val_ssim / (len(val_loader) * val_loader.batch_size)
        
        # Print results
        print(f"Pretraining Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val PSNR: {avg_val_psnr:.2f}dB")
        print(f"  Val SSIM: {avg_val_ssim:.4f}")
        
        # Save if this is the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  New best validation loss: {best_val_loss:.4f}")
            
            # Save the pretrained extraction network
            torch.save({
                'epoch': epoch + 1,
                'extraction_network_state_dict': steg_system.extraction_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_psnr': avg_val_psnr,
                'val_ssim': avg_val_ssim,
            }, 'pretrained_extraction_network.pth')
    
    print("Pretraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load the best model from pretraining
    pretrained_checkpoint = torch.load('pretrained_extraction_network.pth')
    steg_system.extraction_network.load_state_dict(pretrained_checkpoint['extraction_network_state_dict'])
    
    # Return the model to its normal state (all components trainable)
    steg_system.train()


def visualize_results(steg_system, dataloader, device, save_dir, epoch, 
                     use_high_attention=True, num_samples=5):
    """
    Visualize the results of the steganography system
    """
    steg_system.eval()
    
    # Get a batch of data
    cover_imgs, secret_imgs = next(iter(dataloader))
    cover_imgs = cover_imgs.to(device)
    secret_imgs = secret_imgs.to(device)
    
    # Limit to num_samples
    cover_imgs = cover_imgs[:num_samples]
    secret_imgs = secret_imgs[:num_samples]
    
    with torch.no_grad():
        # Get container images and attention maps
        container_imgs, cover_attention, secret_attention, embedding_map = steg_system.forward_hide(
            cover_imgs, secret_imgs, use_high_attention=use_high_attention, return_maps=True
        )
        
        # Extract secrets
        extracted_secrets = steg_system.forward_extract(container_imgs)
        
        # Create visualization grid
        fig, axes = plt.subplots(num_samples, 6, figsize=(20, 4 * num_samples))
        
        # Handle case where there's only one sample
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        attention_mode = "High" if use_high_attention else "Low"
        
        for i in range(num_samples):
            # Convert tensors to numpy for visualization
            cover_np = cover_imgs[i].cpu().numpy().transpose(1, 2, 0)
            secret_np = secret_imgs[i].cpu().numpy().transpose(1, 2, 0)
            container_np = container_imgs[i].cpu().numpy().transpose(1, 2, 0)
            extracted_np = extracted_secrets[i].cpu().numpy().transpose(1, 2, 0)
            
            # Get attention maps
            cover_attn_np = cover_attention[i, 0].cpu().numpy()
            secret_attn_np = secret_attention[i, 0].cpu().numpy()
            
            # Calculate residual (difference map x10 for visibility)
            diff = np.abs(container_np - cover_np) * 10
            
            # Calculate metrics
            psnr_container = calculate_psnr(cover_np, container_np)
            psnr_secret = calculate_psnr(secret_np, extracted_np)
            ssim_container = calculate_ssim(cover_np, container_np)
            ssim_secret = calculate_ssim(secret_np, extracted_np)
            
            # Plot images
            axes[i, 0].imshow(cover_np)
            axes[i, 0].set_title("Cover Image")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(secret_np)
            axes[i, 1].set_title("Secret Image")
            axes[i, 1].axis("off")
            
            # Plot attention maps
            axes[i, 2].imshow(cover_attn_np, cmap='hot')
            axes[i, 2].set_title(f"{attention_mode} Attention Map (Cover)")
            axes[i, 2].axis("off")
            
            axes[i, 3].imshow(container_np)
            axes[i, 3].set_title(f"Container\nPSNR: {psnr_container:.2f}dB, SSIM: {ssim_container:.4f}")
            axes[i, 3].axis("off")
            
            axes[i, 4].imshow(diff)
            axes[i, 4].set_title("Residual (x10)")
            axes[i, 4].axis("off")
            
            axes[i, 5].imshow(extracted_np)
            axes[i, 5].set_title(f"Extracted Secret\nPSNR: {psnr_secret:.2f}dB, SSIM: {ssim_secret:.4f}")
            axes[i, 5].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'epoch_{epoch+1}_{attention_mode}_attention.png'))
        plt.close()


def save_checkpoint(checkpoint_data, filepath):
    """
    Save a checkpoint to disk
    """
    torch.save(checkpoint_data, filepath)
    print(f"Checkpoint saved to {filepath}")


def train_epoch(steg_system, dataloader, optimizer, device, 
                alpha=0.3, beta=0.7, use_high_attention=True, train_both=False):
    """
    Train for one epoch using the enhanced steganography system with attention diversity loss
    """
    steg_system.train()
    
    total_loss = 0.0
    total_hiding_loss = 0.0
    total_extraction_loss = 0.0
    total_attention_loss = 0.0
    
    # For tracking metrics when training with both attention modes
    if train_both:
        opposite_total_loss = 0.0
        opposite_total_hiding_loss = 0.0
        opposite_total_extraction_loss = 0.0
        opposite_total_attention_loss = 0.0
    
    criterion = CombinedLoss(alpha=0.7)
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for i, (cover_imgs, secret_imgs) in enumerate(progress_bar):
        cover_imgs = cover_imgs.to(device)
        secret_imgs = secret_imgs.to(device)
        
        optimizer.zero_grad()
        
        # Process with primary attention mode (high or low based on use_high_attention)
        container_imgs, cover_attention, secret_attention, embedding_map = steg_system.forward_hide(
            cover_imgs, secret_imgs, use_high_attention=use_high_attention
        )
        extracted_secrets = steg_system.forward_extract(container_imgs)
        
        # Calculate standard losses for primary mode
        hiding_loss = criterion(container_imgs, cover_imgs)
        extraction_loss = criterion(extracted_secrets, secret_imgs)
        perceptual_loss = F.l1_loss(extracted_secrets, secret_imgs)
        
        # Add attention diversity loss for primary mode
        cover_std = cover_attention.view(cover_attention.size(0), -1).std(dim=1).mean()
        secret_std = secret_attention.view(secret_attention.size(0), -1).std(dim=1).mean()
        attention_loss = torch.exp(-5 * cover_std) + torch.exp(-5 * secret_std)
        
        # Combined loss for primary mode
        loss = alpha * hiding_loss + beta * (extraction_loss + 0.5 * perceptual_loss) + 0.1 * attention_loss
        
        # Process with opposite attention mode if training both
        if train_both:
            # Use the same cover and secret images but with opposite attention mode
            opposite_container_imgs, opposite_cover_attention, opposite_secret_attention, opposite_embedding_map = steg_system.forward_hide(
                cover_imgs, secret_imgs, use_high_attention=(not use_high_attention)
            )
            opposite_extracted_secrets = steg_system.forward_extract(opposite_container_imgs)
            
            # Calculate losses for opposite mode
            opposite_hiding_loss = criterion(opposite_container_imgs, cover_imgs)
            opposite_extraction_loss = criterion(opposite_extracted_secrets, secret_imgs)
            opposite_perceptual_loss = F.l1_loss(opposite_extracted_secrets, secret_imgs)
            
            # Attention diversity loss for opposite mode
            opposite_cover_std = opposite_cover_attention.view(opposite_cover_attention.size(0), -1).std(dim=1).mean()
            opposite_secret_std = opposite_secret_attention.view(opposite_secret_attention.size(0), -1).std(dim=1).mean()
            opposite_attention_loss = torch.exp(-5 * opposite_cover_std) + torch.exp(-5 * opposite_secret_std)
            
            # Combined loss for opposite mode
            opposite_loss = alpha * opposite_hiding_loss + beta * (opposite_extraction_loss + 0.5 * opposite_perceptual_loss) + 0.1 * opposite_attention_loss
            
            # Backward both losses
            loss.backward()
            opposite_loss.backward()
            
            # Update opposite mode stats
            opposite_total_loss += opposite_loss.item()
            opposite_total_hiding_loss += opposite_hiding_loss.item()
            opposite_total_extraction_loss += opposite_extraction_loss.item()
            opposite_total_attention_loss += opposite_attention_loss.item()
        else:
            # Regular backward pass for single mode
            loss.backward()
        
        optimizer.step()
        
        # Update stats for primary mode
        total_loss += loss.item()
        total_hiding_loss += hiding_loss.item()
        total_extraction_loss += extraction_loss.item()
        total_attention_loss += attention_loss.item()
        
        # Update progress bar
        if train_both:
            progress_bar.set_postfix({
                'loss': f'{(loss.item() + opposite_loss.item()) / 2:.4f}',
                'hiding': f'{(hiding_loss.item() + opposite_hiding_loss.item()) / 2:.4f}',
                'extract': f'{(extraction_loss.item() + opposite_extraction_loss.item()) / 2:.4f}'
            })
        else:
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'hiding': f'{hiding_loss.item():.4f}',
                'extract': f'{extraction_loss.item():.4f}'
            })
    
    # Calculate average losses for primary mode
    avg_loss = total_loss / len(dataloader)
    avg_hiding_loss = total_hiding_loss / len(dataloader)
    avg_extraction_loss = total_extraction_loss / len(dataloader)
    
    results = (avg_loss, avg_hiding_loss, avg_extraction_loss)
    
    # Calculate average losses for opposite mode if applicable
    if train_both:
        opposite_avg_loss = opposite_total_loss / len(dataloader)
        opposite_avg_hiding_loss = opposite_total_hiding_loss / len(dataloader)
        opposite_avg_extraction_loss = opposite_total_extraction_loss / len(dataloader)
        
        opposite_results = (opposite_avg_loss, opposite_avg_hiding_loss, opposite_avg_extraction_loss)
        return results, opposite_results
    
    return results


def validate(steg_system, dataloader, device, alpha=0.5, beta=0.5, use_high_attention=True):
    """
    Validate the model using the enhanced steganography system
    
    Args:
        steg_system: EnhancedSteganographySystem model
        dataloader: DataLoader for the validation set
        device: Device to use (cuda/cpu)
        alpha: Weight for cover-container loss
        beta: Weight for secret-extracted loss
        use_high_attention: Whether to use high attention regions (True) or low attention regions (False)
    
    Returns:
        Validation metrics
    """
    steg_system.eval()
    
    total_loss = 0.0
    total_hiding_loss = 0.0
    total_extraction_loss = 0.0
    total_psnr_container = 0.0
    total_psnr_secret = 0.0
    total_ssim_container = 0.0
    total_ssim_secret = 0.0
    
    criterion = CombinedLoss(alpha=0.7)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validating')
        
        for i, (cover_imgs, secret_imgs) in enumerate(progress_bar):
            cover_imgs = cover_imgs.to(device)
            secret_imgs = secret_imgs.to(device)
            
            # Forward pass using the enhanced system
            container_imgs, cover_atn, secret_atn, embed_map = steg_system.forward_hide(
                cover_imgs, secret_imgs, use_high_attention=use_high_attention
            )
            extracted_secrets = steg_system.forward_extract(container_imgs)
            
            # Calculate losses
            hiding_loss = criterion(container_imgs, cover_imgs)
            extraction_loss = criterion(extracted_secrets, secret_imgs)
            
            # Combined loss
            loss = alpha * hiding_loss + beta * extraction_loss
            
            # Calculate metrics
            for j in range(cover_imgs.size(0)):
                # Convert to numpy for PSNR and SSIM calculation
                cover_np = cover_imgs[j].cpu().numpy().transpose(1, 2, 0)
                container_np = container_imgs[j].cpu().numpy().transpose(1, 2, 0)
                secret_np = secret_imgs[j].cpu().numpy().transpose(1, 2, 0)
                extracted_np = extracted_secrets[j].cpu().numpy().transpose(1, 2, 0)
                
                # Calculate PSNR (normalized [0,1] images)
                psnr_container = calculate_psnr(cover_np, container_np)
                psnr_secret = calculate_psnr(secret_np, extracted_np)
                
                # Calculate SSIM (normalized [0,1] images)
                ssim_container = calculate_ssim(cover_np, container_np)
                ssim_secret = calculate_ssim(secret_np, extracted_np)
                
                total_psnr_container += psnr_container
                total_psnr_secret += psnr_secret
                total_ssim_container += ssim_container
                total_ssim_secret += ssim_secret
            
            # Update stats
            total_loss += loss.item()
            total_hiding_loss += hiding_loss.item()
            total_extraction_loss += extraction_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr_c': f'{psnr_container:.2f}',
                'psnr_s': f'{psnr_secret:.2f}'
            })
    
    # Calculate averages
    n_samples = len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    avg_hiding_loss = total_hiding_loss / len(dataloader)
    avg_extraction_loss = total_extraction_loss / len(dataloader)
    avg_psnr_container = total_psnr_container / n_samples
    avg_psnr_secret = total_psnr_secret / n_samples
    avg_ssim_container = total_ssim_container / n_samples
    avg_ssim_secret = total_ssim_secret / n_samples
    
    return (avg_loss, avg_hiding_loss, avg_extraction_loss, 
            avg_psnr_container, avg_psnr_secret, 
            avg_ssim_container, avg_ssim_secret)

def pretrain_extraction(steg_system, train_loader, val_loader, device, num_epochs=5, lr=0.0002):
    """
    Pretrain the extraction network alone to improve its base capabilities
    before joint training with the hiding network.
    
    Args:
        steg_system: The complete steganography system
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to run training on
        num_epochs: Number of pretraining epochs
        lr: Learning rate for pretraining
    """
    print("Starting extraction network pretraining...")
    
    # We'll only train the extraction network
    # First, set all networks to eval mode
    steg_system.eval()
    
    # Then set extraction network to train mode
    steg_system.extraction_network.train()
    
    # Define optimizer just for extraction network parameters
    optimizer = optim.AdamW(steg_system.extraction_network.parameters(), lr=lr)
    
    # Define criterion (combined loss)
    criterion = CombinedLoss(alpha=1.0)  # Full weight on reconstruction quality
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        steg_system.extraction_network.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Pretraining Epoch {epoch+1}/{num_epochs}')
        
        for i, (_, secret_imgs) in enumerate(progress_bar):
            # We only need the secret images for pretraining the extractor
            secret_imgs = secret_imgs.to(device)
            
            optimizer.zero_grad()
            
            # Generate noisy container images (simulating the output of hiding network)
            # This helps the extraction network learn to be robust to various distortions
            with torch.no_grad():
                # First get container images with the current hiding network
                container_imgs, _, _, _ = steg_system.forward_hide(
                    secret_imgs, secret_imgs, use_high_attention=True
                )
                
                # Add some noise to create more challenging examples
                noise_level = 0.05 * (1.0 - epoch / num_epochs)  # Reduce noise over time
                noise = torch.randn_like(container_imgs) * noise_level
                noisy_containers = torch.clamp(container_imgs + noise, 0, 1)
            
            # Extract the secrets from the noisy containers
            extracted_secrets = steg_system.extraction_network(noisy_containers)
            
            # Calculate loss
            loss = criterion(extracted_secrets, secret_imgs)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Update stats
            train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        steg_system.extraction_network.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        
        with torch.no_grad():
            for i, (_, secret_imgs) in enumerate(val_loader):
                secret_imgs = secret_imgs.to(device)
                
                # Create containers with hiding network
                container_imgs, _, _, _ = steg_system.forward_hide(
                    secret_imgs, secret_imgs, use_high_attention=True
                )
                
                # Extract the secrets
                extracted_secrets = steg_system.extraction_network(container_imgs)
                
                # Calculate metrics
                loss = criterion(extracted_secrets, secret_imgs)
                val_loss += loss.item()
                
                # Calculate PSNR and SSIM for batch
                for j in range(secret_imgs.size(0)):
                    original = secret_imgs[j].cpu().numpy().transpose(1, 2, 0)
                    extracted = extracted_secrets[j].cpu().numpy().transpose(1, 2, 0)
                    
                    val_psnr += calculate_psnr(original, extracted)
                    val_ssim += calculate_ssim(original, extracted)
        
        # Calculate averages
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / (len(val_loader) * val_loader.batch_size)
        avg_val_ssim = val_ssim / (len(val_loader) * val_loader.batch_size)
        
        # Print results
        print(f"Pretraining Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val PSNR: {avg_val_psnr:.2f}dB")
        print(f"  Val SSIM: {avg_val_ssim:.4f}")
        
        # Save if this is the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  New best validation loss: {best_val_loss:.4f}")
            
            # Save the pretrained extraction network
            torch.save({
                'epoch': epoch + 1,
                'extraction_network_state_dict': steg_system.extraction_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_psnr': avg_val_psnr,
                'val_ssim': avg_val_ssim,
            }, 'pretrained_extraction_network.pth')
    
    print("Pretraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load the best model from pretraining
    pretrained_checkpoint = torch.load('pretrained_extraction_network.pth')
    steg_system.extraction_network.load_state_dict(pretrained_checkpoint['extraction_network_state_dict'])
    
    # Return the model to its normal state (all components trainable)
    steg_system.train()


def visualize_results(steg_system, dataloader, device, save_dir, epoch, use_high_attention=True):
    """
    Visualize the results on a few sample images using the enhanced steganography system
    """
    steg_system.eval()
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get a batch of images
    cover_imgs, secret_imgs = next(iter(dataloader))
    
    # Determine how many samples to visualize (up to 4, but limited by batch size)
    n_samples = min(4, cover_imgs.size(0))
    
    # Use only the first n_samples images for visualization
    cover_imgs = cover_imgs[:n_samples].to(device)
    secret_imgs = secret_imgs[:n_samples].to(device)
    
    with torch.no_grad():
        # Get the comparison results
        results = steg_system.compare_attention_methods(cover_imgs, secret_imgs)
        
        # Extract the relevant tensors for visualization
        container_high = results["container_high"]
        container_low = results["container_low"]
        extracted_high = results["extracted_high"]
        extracted_low = results["extracted_low"]
        cover_attention = results["cover_attention"]
        secret_attention = results["secret_attention"]
        embedding_map_high = results["embedding_map_high"]
        embedding_map_low = results["embedding_map_low"]
        metrics = results["metrics"]
    
    attention_mode = "high" if use_high_attention else "low"
    
    # Create a figure with n_samples rows and 8 columns for comprehensive visualization
    fig, axes = plt.subplots(n_samples, 8, figsize=(24, 5 * n_samples))
    
    # Handle case where n_samples is 1 (axes would be 1D)
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Extract the data for this sample
        cover_np = cover_imgs[i].cpu().numpy().transpose(1, 2, 0)
        secret_np = secret_imgs[i].cpu().numpy().transpose(1, 2, 0)
        
        # Get the high attention results
        container_high_np = container_high[i].cpu().numpy().transpose(1, 2, 0)
        extracted_high_np = extracted_high[i].cpu().numpy().transpose(1, 2, 0)
        
        # Get the low attention results
        container_low_np = container_low[i].cpu().numpy().transpose(1, 2, 0)
        extracted_low_np = extracted_low[i].cpu().numpy().transpose(1, 2, 0)
        
        # Calculate metrics for this sample
        high_psnr_container = calculate_psnr(cover_np, container_high_np)
        high_ssim_container = calculate_ssim(cover_np, container_high_np)
        high_psnr_secret = calculate_psnr(secret_np, extracted_high_np)
        high_ssim_secret = calculate_ssim(secret_np, extracted_high_np)
        
        low_psnr_container = calculate_psnr(cover_np, container_low_np)
        low_ssim_container = calculate_ssim(cover_np, container_low_np)
        low_psnr_secret = calculate_psnr(secret_np, extracted_low_np)
        low_ssim_secret = calculate_ssim(secret_np, extracted_low_np)
        
        # Plot the images
        axes[i, 0].imshow(np.clip(cover_np, 0, 1))
        axes[i, 0].set_title("Cover Image")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(np.clip(secret_np, 0, 1))
        axes[i, 1].set_title("Secret Image")
        axes[i, 1].axis("off")
        
        # Display attention heatmaps
        axes[i, 2].imshow(cover_attention[i].cpu().numpy().squeeze(), cmap='hot')
        axes[i, 2].set_title("Cover Attention")
        axes[i, 2].axis("off")
        
        axes[i, 3].imshow(secret_attention[i].cpu().numpy().squeeze(), cmap='hot')
        axes[i, 3].set_title("Secret Attention")
        axes[i, 3].axis("off")
        
        # Display embedding maps
        axes[i, 4].imshow(embedding_map_high[i].cpu().numpy().squeeze(), cmap='viridis')
        axes[i, 4].set_title("High Attn Embedding")
        axes[i, 4].axis("off")
        
        axes[i, 5].imshow(embedding_map_low[i].cpu().numpy().squeeze(), cmap='viridis')
        axes[i, 5].set_title("Low Attn Embedding")
        axes[i, 5].axis("off")
        
        # Calculate PSNR and SSIM for each sample
        psnr_high_container = calculate_psnr(cover_np, container_high_np)
        ssim_high_container = calculate_ssim(cover_np, container_high_np)
        psnr_high_secret = calculate_psnr(secret_np, extracted_high_np)
        ssim_high_secret = calculate_ssim(secret_np, extracted_high_np)
        
        psnr_low_container = calculate_psnr(cover_np, container_low_np)
        ssim_low_container = calculate_ssim(cover_np, container_low_np)
        psnr_low_secret = calculate_psnr(secret_np, extracted_low_np)
        ssim_low_secret = calculate_ssim(secret_np, extracted_low_np)
        
        # Display high attention results
        if use_high_attention:
            container_np = container_high_np
            extracted_np = extracted_high_np
            
            axes[i, 6].imshow(np.clip(container_np, 0, 1))
            axes[i, 6].set_title(f'Container (High)\nPSNR: {psnr_high_container:.2f}dB\nSSIM: {ssim_high_container:.4f}')
            axes[i, 6].axis("off")
            
            axes[i, 7].imshow(np.clip(extracted_np, 0, 1))
            axes[i, 7].set_title(f'Extracted (High)\nPSNR: {psnr_high_secret:.2f}dB\nSSIM: {ssim_high_secret:.4f}')
            axes[i, 7].axis("off")
        else:
            container_np = container_low_np
            extracted_np = extracted_low_np
            
            axes[i, 6].imshow(np.clip(container_np, 0, 1))
            axes[i, 6].set_title(f'Container (Low)\nPSNR: {psnr_low_container:.2f}dB\nSSIM: {ssim_low_container:.4f}')
            axes[i, 6].axis("off")
            
            axes[i, 7].imshow(np.clip(extracted_np, 0, 1))
            axes[i, 7].set_title(f'Extracted (Low)\nPSNR: {psnr_low_secret:.2f}dB\nSSIM: {ssim_low_secret:.4f}')
            axes[i, 7].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'results_epoch_{epoch}_{attention_mode}_attention.png'))
    plt.close()
    

def save_checkpoint(state, filename):
    """
    Save checkpoint
    
    Args:
        state: State dictionary
        filename: Filename to save the checkpoint
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Dual-Attention Steganography System on Tiny-ImageNet')
    parser.add_argument('--data_dir', type=str, default='./', help='Path to tiny-imagenet-200 directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=144, help='Image size')
    parser.add_argument('--alpha', type=float, default=0.7, help='Weight for container quality (hiding loss)')
    parser.add_argument('--beta', type=float, default=0.3, help='Weight for secret recovery (extraction loss)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--vis_dir', type=str, default='./visualizations', help='Directory to save visualizations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for SWinT')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, nargs='+', default=[8, 8, 8, 8], help='Number of heads in each layer')
    parser.add_argument('--depths', type=int, nargs='+', default=[6, 6, 6, 6], help='Depth of each layer')
    parser.add_argument('--use_high_attention', type=bool, default=True, help='Whether to use high attention regions (True) or low attention regions (False)')
    parser.add_argument('--train_both', type=bool, default=True, help='Whether to train with both high and low attention modes')
    parser.add_argument('--pretrain', action='store_true', help='Pretrain the extraction network')
    parser.add_argument('--pretrain_epochs', type=int, default=5, help='Number of epochs for pretraining')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    
    # Create datasets and dataloaders
    train_dataset = TinyImageNetDataset(args.data_dir, split='train', transform=transform)
    val_dataset = TinyImageNetDataset(args.data_dir, split='val', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create the enhanced steganography system
    steg_system = EnhancedSteganographySystem(
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        depths=args.depths,
        num_heads=args.num_heads,
        window_size=args.window_size
    ).to(device)
    
    print(f"Created EnhancedSteganographySystem with:")
    print(f"  - Image size: {args.img_size}x{args.img_size}")
    print(f"  - Embedding dimension: {args.embed_dim}")
    print(f"  - Window size: {args.window_size}")
    print(f"  - Depths: {args.depths}")
    print(f"  - Number of heads: {args.num_heads}")
    
    # Define optimizer with parameter groups for different learning rates
    params_dict = [
        {"params": steg_system.cover_attention.parameters(), "lr": args.lr * 0.5},
        {"params": steg_system.secret_attention.parameters(), "lr": args.lr * 0.5},
        {"params": steg_system.hiding_network.parameters(), "lr": args.lr},
        {"params": steg_system.extraction_network.parameters(), "lr": args.lr},
    ]
    
    optimizer = optim.AdamW(params_dict, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Initialize variables
    start_epoch = 0
    best_val_loss = float('inf')
    best_container_psnr = 0
    best_secret_psnr = 0
    
    # Resume from checkpoint if specified
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_container_psnr = checkpoint.get('best_container_psnr', 0)
        best_secret_psnr = checkpoint.get('best_secret_psnr', 0)
        
        steg_system.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
    
    # Pretraining stage if enabled
    if args.pretrain:
        print(f"\nStarting pretraining for {args.pretrain_epochs} epochs...")
        pretrain_extraction(
            steg_system, 
            train_loader, 
            val_loader, 
            device, 
            num_epochs=args.pretrain_epochs,
            lr=args.lr * 2  # Higher learning rate for pretraining
        )
    
    # Training loop with curriculum learning
    print("Starting main training...")
    for epoch in range(start_epoch, args.epochs):
        # Calculate dynamic embedding parameters based on epoch
        # Start with emphasis on extraction, gradually balance with hiding
        progress_ratio = min(1.0, epoch / (args.epochs * 0.5))  # First half of training
        
        # Dynamically adjust the alpha and beta parameters
        current_alpha = min(0.7, args.alpha + (0.2 * progress_ratio))  # Start lower, increase to target
        current_beta = max(0.3, args.beta - (0.2 * progress_ratio))   # Start higher, decrease to target
        
        print(f"\nEpoch {epoch+1}/{args.epochs} (alpha={current_alpha:.2f}, beta={current_beta:.2f})")
        
        # Train with specified attention mode using curriculum learning parameters
        train_result = train_epoch(
            steg_system, train_loader, optimizer, device, 
            alpha=current_alpha, beta=current_beta, 
            use_high_attention=args.use_high_attention, train_both=args.train_both
        )
        
        # Handle the return values based on train_both mode
        if args.train_both:
            # If training both modes, we get two result tuples
            primary_results, opposite_results = train_result
            train_loss, train_hiding_loss, train_extraction_loss = primary_results
            # You can also use opposite_results if needed
        else:
            # If training single mode, we get one result tuple
            train_loss, train_hiding_loss, train_extraction_loss = train_result
        
        # Validate with specified attention mode
        val_metrics = validate(
            steg_system, val_loader, device, 
            alpha=current_alpha, beta=current_beta, 
            use_high_attention=args.use_high_attention
        )
        
        val_loss, val_hiding_loss, val_extraction_loss, val_psnr_container, val_psnr_secret, val_ssim_container, val_ssim_secret = val_metrics
        
        # Visualize results with specified attention mode
        visualize_results(steg_system, val_loader, device, args.vis_dir, epoch, use_high_attention=args.use_high_attention)
        
        # If train_both flag is set, also validate and visualize with the opposite attention mode
        if args.train_both:
            print(f"Validating with {'low' if args.use_high_attention else 'high'} attention regions...")
            opposite_val_metrics = validate(
                steg_system, val_loader, device, 
                alpha=current_alpha, beta=current_beta, 
                use_high_attention=not args.use_high_attention
            )
            
            opposite_val_loss, opposite_val_hiding_loss, opposite_val_extraction_loss, opposite_val_psnr_container, opposite_val_psnr_secret, opposite_val_ssim_container, opposite_val_ssim_secret = opposite_val_metrics
            
            # Visualize results with opposite attention mode
            visualize_results(steg_system, val_loader, device, args.vis_dir, epoch, use_high_attention=not args.use_high_attention)
        
        # Step the scheduler
        scheduler.step()
        
        # Print metrics for specified attention mode
        attention_mode = "high" if args.use_high_attention else "low"
        print(f"\n{attention_mode.capitalize()} Attention Mode Results:")
        print(f"Train Loss: {train_loss:.4f}, Hiding Loss: {train_hiding_loss:.4f}, Extraction Loss: {train_extraction_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Hiding Loss: {val_hiding_loss:.4f}, Extraction Loss: {val_extraction_loss:.4f}")
        print(f"PSNR Container: {val_psnr_container:.2f}dB, PSNR Secret: {val_psnr_secret:.2f}dB")
        print(f"SSIM Container: {val_ssim_container:.4f}, SSIM Secret: {val_ssim_secret:.4f}")
        
        # Print metrics for opposite attention mode if applicable
        if args.train_both:
            opposite_mode = "low" if args.use_high_attention else "high"
            print(f"\n{opposite_mode.capitalize()} Attention Mode Results:")
            print(f"Train Loss: {train_loss:.4f}, Hiding Loss: {train_hiding_loss:.4f}, Extraction Loss: {train_extraction_loss:.4f}")
            print(f"Val Loss: {opposite_val_loss:.4f}, Hiding Loss: {opposite_val_hiding_loss:.4f}, Extraction Loss: {opposite_val_extraction_loss:.4f}")
            print(f"PSNR Container: {opposite_val_psnr_container:.2f}dB, PSNR Secret: {opposite_val_psnr_secret:.2f}dB")
            print(f"SSIM Container: {opposite_val_ssim_container:.4f}, SSIM Secret: {opposite_val_ssim_secret:.4f}")
        
        # Save checkpoint for each epoch
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': steg_system.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr_container': val_psnr_container,
            'val_psnr_secret': val_psnr_secret,
            'val_ssim_container': val_ssim_container,
            'val_ssim_secret': val_ssim_secret,
            'best_val_loss': best_val_loss,
            'best_container_psnr': best_container_psnr,
            'best_secret_psnr': best_secret_psnr,
            'attention_mode': attention_mode,
        }
        
        save_checkpoint(
            checkpoint_data, 
            os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}_{attention_mode}_attention.pth')
        )
        
        # Check if this is the best model by container PSNR (primary metric)
        is_best_container = val_psnr_container > best_container_psnr
        if is_best_container:
            best_container_psnr = val_psnr_container
            save_checkpoint(
                checkpoint_data,
                os.path.join(args.save_dir, f'best_container_model_{attention_mode}_attention.pth')
            )
            
        # Also save best model by secret PSNR
        is_best_secret = val_psnr_secret > best_secret_psnr
        if is_best_secret:
            best_secret_psnr = val_psnr_secret
            save_checkpoint(
                checkpoint_data,
                os.path.join(args.save_dir, f'best_secret_model_{attention_mode}_attention.pth')
            )
            
        # If training both modes, also save the best from opposite mode
        if args.train_both:
            opposite_mode = "low" if args.use_high_attention else "high"
            
            # Save best opposite container model
            if opposite_val_psnr_container > best_container_psnr:
                best_container_psnr = opposite_val_psnr_container
                opposite_checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': steg_system.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': opposite_val_loss,
                    'val_psnr_container': opposite_val_psnr_container,
                    'val_psnr_secret': opposite_val_psnr_secret,
                    'val_ssim_container': opposite_val_ssim_container,
                    'val_ssim_secret': opposite_val_ssim_secret,
                    'best_val_loss': best_val_loss,
                    'best_container_psnr': best_container_psnr,
                    'best_secret_psnr': best_secret_psnr,
                    'attention_mode': opposite_mode,
                }
                save_checkpoint(
                    opposite_checkpoint_data,
                    os.path.join(args.save_dir, f'best_container_model_{opposite_mode}_attention.pth')
                )
    
    print("Training completed!")
    
    # Final evaluation with both attention modes
    print("\nFinal Evaluation:")
    
    # Create a test batch for final evaluation
    final_val_loader = DataLoader(val_dataset, batch_size=min(16, len(val_dataset)), shuffle=True, num_workers=4, pin_memory=True)
    test_cover, test_secret = next(iter(final_val_loader))
    test_cover, test_secret = test_cover.to(device), test_secret.to(device)
    
    # Run final comprehensive evaluation with visualizations
    with torch.no_grad():
        # Get results for both modes
        results = steg_system.compare_attention_methods(test_cover, test_secret)
        metrics = results["metrics"]
        
        # Create the final comparison visualization
        fig, axes = plt.subplots(len(test_cover), 5, figsize=(20, 4 * len(test_cover)))
        
        # Handle case where there's only one sample
        if len(test_cover) == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(len(test_cover)):
            # Extract the data for this sample
            cover_np = test_cover[i].cpu().numpy().transpose(1, 2, 0)
            secret_np = test_secret[i].cpu().numpy().transpose(1, 2, 0)
            
            # Get the high attention results
            container_high_np = results["container_high"][i].cpu().numpy().transpose(1, 2, 0)
            extracted_high_np = results["extracted_high"][i].cpu().numpy().transpose(1, 2, 0)
            
            # Get the low attention results
            container_low_np = results["container_low"][i].cpu().numpy().transpose(1, 2, 0)
            extracted_low_np = results["extracted_low"][i].cpu().numpy().transpose(1, 2, 0)
            
            # Calculate metrics for this sample
            high_psnr_container = calculate_psnr(cover_np, container_high_np)
            high_ssim_container = calculate_ssim(cover_np, container_high_np)
            high_psnr_secret = calculate_psnr(secret_np, extracted_high_np)
            high_ssim_secret = calculate_ssim(secret_np, extracted_high_np)
            
            low_psnr_container = calculate_psnr(cover_np, container_low_np)
            low_ssim_container = calculate_ssim(cover_np, container_low_np)
            low_psnr_secret = calculate_psnr(secret_np, extracted_low_np)
            low_ssim_secret = calculate_ssim(secret_np, extracted_low_np)
            
            # Plot the images
            axes[i, 0].imshow(cover_np)
            axes[i, 0].set_title("Cover Image")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(secret_np)
            axes[i, 1].set_title("Secret Image")
            axes[i, 1].axis("off")
            
            axes[i, 2].imshow(container_high_np)
            axes[i, 2].set_title(f"High Attention Container\nPSNR: {high_psnr_container:.2f}, SSIM: {high_ssim_container:.4f}")
            axes[i, 2].axis("off")
            
            axes[i, 3].imshow(container_low_np)
            axes[i, 3].set_title(f"Low Attention Container\nPSNR: {low_psnr_container:.2f}, SSIM: {low_ssim_container:.4f}")
            axes[i, 3].axis("off")
            
            # Show the better extraction result based on PSNR
            if high_psnr_secret > low_psnr_secret:
                axes[i, 4].imshow(extracted_high_np)
                axes[i, 4].set_title(f"Best Extracted Secret (High)\nPSNR: {high_psnr_secret:.2f}, SSIM: {high_ssim_secret:.4f}")
            else:
                axes[i, 4].imshow(extracted_low_np)
                axes[i, 4].set_title(f"Best Extracted Secret (Low)\nPSNR: {low_psnr_secret:.2f}, SSIM: {low_ssim_secret:.4f}")
            axes[i, 4].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.vis_dir, "final_comparison.png"))
        
        # Print the average metrics
        print("\nAverage Results Across Test Batch:")
        print("High Attention Mode:")
        print(f"  Container - PSNR: {metrics['high_attention']['container']['psnr']:.2f}dB, SSIM: {metrics['high_attention']['container']['ssim']:.4f}")
        print(f"  Extracted - PSNR: {metrics['high_attention']['extracted']['psnr']:.2f}dB, SSIM: {metrics['high_attention']['extracted']['ssim']:.4f}")
        print("Low Attention Mode:")
        print(f"  Container - PSNR: {metrics['low_attention']['container']['psnr']:.2f}dB, SSIM: {metrics['low_attention']['container']['ssim']:.4f}")
        print(f"  Extracted - PSNR: {metrics['low_attention']['extracted']['psnr']:.2f}dB, SSIM: {metrics['low_attention']['extracted']['ssim']:.4f}")
        
        # Determine which mode performed better overall
        high_total = metrics['high_attention']['container']['psnr'] + metrics['high_attention']['extracted']['psnr']
        low_total = metrics['low_attention']['container']['psnr'] + metrics['low_attention']['extracted']['psnr']
        
        if high_total > low_total:
            print("\n High attention embedding performed better overall")
        else:
            print("\n Low attention embedding performed better overall")
        
        # Give advice for real-world usage
        print("\nRecommendation for optimal usage:")
        if metrics['high_attention']['container']['psnr'] > metrics['low_attention']['container']['psnr']:
            print("- For best container quality (imperceptibility): Use HIGH attention embedding")
        else:
            print("- For best container quality (imperceptibility): Use LOW attention embedding")
            
        if metrics['high_attention']['extracted']['psnr'] > metrics['low_attention']['extracted']['psnr']:
            print("- For best secret recovery: Use HIGH attention embedding")
        else:
            print("- For best secret recovery: Use LOW attention embedding")

if __name__ == "__main__":
    main()

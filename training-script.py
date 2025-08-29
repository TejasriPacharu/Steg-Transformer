#set seed
#tinyimagenetdataset class
#combined loss class
# calculate psnr function
# calculate ssim function
# validate function
# pretrain extraction function
# visualize results function with samples
# save checkpoint function
# train epoch function
# visualize results function 
# save checkpoint for state and file name

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

from steg_system import EnhancedSteganographySystem

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
        try:
            # Pick random index for cover
            cover_idx = random.randint(0, len(self.img_paths) - 1)
            cover_image = Image.open(self.img_paths[cover_idx]).convert('RGB')

            # Pick random index for secret (different from cover)
            secret_idx = random.randint(0, len(self.img_paths) - 1)
            while secret_idx == cover_idx:
                secret_idx = random.randint(0, len(self.img_paths) - 1)

            secret_image = Image.open(self.img_paths[secret_idx]).convert('RGB')

            if self.transform:
                cover_image = self.transform(cover_image)
                secret_image = self.transform(secret_image)

            return cover_image, secret_image

        except Exception as e:
            print(f"Error loading image: {e}")
            if self.transform:
                return (
                    self.transform(Image.new('RGB', (64, 64), (0, 0, 0))),
                    self.transform(Image.new('RGB', (64, 64), (0, 0, 0)))
                )
            else:
                return (
                    Image.new('RGB', (64, 64), (0, 0, 0)),
                    Image.new('RGB', (64, 64), (0, 0, 0))
                )


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


def validate(steg_system, dataloader, device, alpha=0.3, beta=0.7, use_high_attention=True):
    """
    Validate the model on the validation set with enhanced metrics including attention diversity
    """
    steg_system.eval()
    
    val_loss = 0.0
    val_hiding_loss = 0.0
    val_extraction_loss = 0.0
    val_attention_loss = 0.0
    
    # For tracking container and secret image quality
    total_psnr_container = 0.0
    total_ssim_container = 0.0
    total_psnr_secret = 0.0
    total_ssim_secret = 0.0
    
    criterion = CombinedLoss(alpha=0.7)
    
    with torch.no_grad():
        for cover_imgs, secret_imgs in tqdm(dataloader, desc="Validating"):
            cover_imgs = cover_imgs.to(device)
            secret_imgs = secret_imgs.to(device)
            
            # Forward pass using the enhanced system
            container_imgs, cover_attn, secret_attn, embed_map = steg_system.forward_hide(
                cover_imgs, secret_imgs, use_high_attention=use_high_attention
            )
            extracted_secrets = steg_system.forward_extract(container_imgs)
            
            # Calculate losses
            hiding_loss = criterion(container_imgs, cover_imgs)
            extraction_loss = criterion(extracted_secrets, secret_imgs)
            
            # Calculate attention diversity loss
            attn_loss = steg_system.attention_loss(cover_attn, secret_attn)
            
            # Combined loss with attention diversity component
            attn_weight = 0.1  # Keep consistent with training
            loss = alpha * hiding_loss + beta * extraction_loss + attn_weight * attn_loss
            
            # Update validation metrics
            val_loss += loss.item()
            val_hiding_loss += hiding_loss.item()
            val_extraction_loss += extraction_loss.item()
            val_attention_loss += attn_loss.item()
            
            # Calculate PSNR and SSIM metrics
            for i in range(cover_imgs.size(0)):
                cover_np = cover_imgs[i].cpu().permute(1, 2, 0).numpy()
                secret_np = secret_imgs[i].cpu().permute(1, 2, 0).numpy()
                container_np = container_imgs[i].cpu().permute(1, 2, 0).numpy()
                extracted_np = extracted_secrets[i].cpu().permute(1, 2, 0).numpy()
                
                # PSNR for container (cover vs container)
                psnr_container = calculate_psnr(cover_np, container_np)
                total_psnr_container += psnr_container
                
                # SSIM for container (cover vs container)
                ssim_container = calculate_ssim(cover_np, container_np)
                total_ssim_container += ssim_container
                
                # PSNR for secret (original secret vs extracted)
                psnr_secret = calculate_psnr(secret_np, extracted_np)
                total_psnr_secret += psnr_secret
                
                # SSIM for secret (original secret vs extracted)
                ssim_secret = calculate_ssim(secret_np, extracted_np)
                total_ssim_secret += ssim_secret
    
    # Calculate average metrics
    num_samples = len(dataloader.dataset)
    avg_val_loss = val_loss / len(dataloader)
    avg_val_hiding_loss = val_hiding_loss / len(dataloader)
    avg_val_extraction_loss = val_extraction_loss / len(dataloader)
    avg_val_attention_loss = val_attention_loss / len(dataloader)
    
    avg_psnr_container = total_psnr_container / num_samples
    avg_ssim_container = total_ssim_container / num_samples
    avg_psnr_secret = total_psnr_secret / num_samples
    avg_ssim_secret = total_ssim_secret / num_samples
    
    # Print validation results with attention metrics
    print(f"Validation - Loss: {avg_val_loss:.4f} (Hide: {avg_val_hiding_loss:.4f}, Extract: {avg_val_extraction_loss:.4f}, Attn: {avg_val_attention_loss:.4f})")
    print(f"Validation - Container: PSNR={avg_psnr_container:.2f}dB, SSIM={avg_ssim_container:.4f}")
    print(f"Validation - Secret: PSNR={avg_psnr_secret:.2f}dB, SSIM={avg_ssim_secret:.4f}")
    
    return (avg_val_loss, avg_val_hiding_loss, avg_val_extraction_loss, 
            avg_psnr_container, avg_psnr_secret, avg_ssim_container, avg_ssim_secret, 
            avg_val_attention_loss)


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
    avg_attention_loss = total_attention_loss / len(dataloader)

    results = (avg_loss, avg_hiding_loss, avg_extraction_loss, avg_attention_loss)

    # Calculate average losses for opposite mode if applicable
    if train_both:
        opposite_avg_loss = opposite_total_loss / len(dataloader)
        opposite_avg_hiding_loss = opposite_total_hiding_loss / len(dataloader)
        opposite_avg_extraction_loss = opposite_total_extraction_loss / len(dataloader)
        opposite_avg_attention_loss = opposite_total_attention_loss / len(dataloader)
        
        opposite_results = (opposite_avg_loss, opposite_avg_hiding_loss, opposite_avg_extraction_loss, opposite_avg_attention_loss)
        return results, opposite_results
    
    return results

def select_visualization_samples(dataloader, device):
    """
    Select a random set of samples for visualization during the current epoch.
    These samples will be used for both high and low attention embedding visualization.
    A new set of samples will be selected for the next epoch.
    
    Args:
        dataloader: DataLoader containing the dataset
        device: Device to store tensors on
        
    Returns:
        Tuple of (cover_imgs, secret_imgs) to use for visualization
    """
    # Get a batch of images from the dataloader
    cover_imgs, secret_imgs = next(iter(dataloader))
    
    # Determine how many samples to visualize (up to 4, but limited by batch size)
    n_samples = min(4, cover_imgs.size(0))
    
    # Use only the first n_samples images for visualization
    cover_imgs = cover_imgs[:n_samples].to(device)
    secret_imgs = secret_imgs[:n_samples].to(device)
    
    return cover_imgs, secret_imgs

def visualize_results(steg_system, dataloader, device, save_dir, epoch):
    """
    Visualize the results on a few sample images using the enhanced steganography system.
    For each epoch, randomly select cover and secret images, and use the same images
    for both high and low attention visualization.
    """
    # Set model to evaluation mode to ensure deterministic behavior
    steg_system.eval()
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Select random samples for this epoch's visualization
    cover_imgs, secret_imgs = select_visualization_samples(dataloader, device)
    
    # Determine how many samples we have
    n_samples = cover_imgs.size(0)
    
    with torch.no_grad():
        # Get the comparison results - this will generate results for both
        # high and low attention using the same cover and secret images
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
    
    # Create figures for both high and low attention results
    for attention_mode in ['high', 'low']:
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
            
            # Display container and extracted image based on attention mode
            if attention_mode == 'high':
                container_np = container_high_np
                extracted_np = extracted_high_np
                psnr_container = high_psnr_container
                ssim_container = high_ssim_container
                psnr_secret = high_psnr_secret
                ssim_secret = high_ssim_secret
            else:
                container_np = container_low_np
                extracted_np = extracted_low_np
                psnr_container = low_psnr_container
                ssim_container = low_ssim_container
                psnr_secret = low_psnr_secret
                ssim_secret = low_ssim_secret
                
            axes[i, 6].imshow(np.clip(container_np, 0, 1))
            axes[i, 6].set_title(f'Container ({attention_mode.capitalize()})\nPSNR: {psnr_container:.2f}dB\nSSIM: {ssim_container:.4f}')
            axes[i, 6].axis("off")
            
            axes[i, 7].imshow(np.clip(extracted_np, 0, 1))
            axes[i, 7].set_title(f'Extracted ({attention_mode.capitalize()})\nPSNR: {psnr_secret:.2f}dB\nSSIM: {ssim_secret:.4f}')
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
    parser = argparse.ArgumentParser(description="Train the Enhanced Swin-Transformer Steganography System")
    
    # Dataset arguments
    parser.add_argument('--dataset_root', type=str, default='./tiny-imagenet-200', help='Dataset root directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--img_size', type=int, default=144, help='Image size for training')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for Swin Transformer')
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained model if available')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.3, help='Final weight for hiding loss')
    parser.add_argument('--beta', type=float, default=0.7, help='Final weight for extraction loss')
    parser.add_argument('--use_high_attention', action='store_true', default=True, help='Use high attention areas for embedding')
    parser.add_argument('--pretrain_extractor', action='store_true', help='Pretrain the extraction network')
    parser.add_argument('--pretrain_epochs', type=int, default=10, help='Number of pretraining epochs')
    parser.add_argument('--train_both', action='store_true', help='Train with both high and low attention methods')
    
    # Other arguments
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.save_dir}/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    
    # Save configuration
    config_dict = vars(args)
    config_path = os.path.join(save_dir, "config.txt")
    with open(config_path, "w") as f:
        for k, v in config_dict.items():
            f.write(f"{k}: {v}\n")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    train_dataset = TinyImageNetDataset(
        root_dir=args.dataset_root, split='train', transform=transform
    )
    val_dataset = TinyImageNetDataset(
        root_dir=args.dataset_root, split='val', transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Create model
    print("Initializing Enhanced Steganography System...")
    steg_system = EnhancedSteganographySystem(
        img_size=args.img_size, embed_dim=args.embed_dim, 
        window_size=args.window_size
    ).to(device)
    
    # Print model details
    print(f"Model parameters: {sum(p.numel() for p in steg_system.parameters() if p.requires_grad)}")
    
    # Define optimizer with improved parameters
    optimizer = torch.optim.AdamW(
        steg_system.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.99),  # Improved stability
        eps=1e-8,
        weight_decay=1e-4  # Regularization to prevent overfitting
    )
    
    # Define learning rate scheduler with warmup and cosine decay
    # This helps with training stability and convergence
    def warmup_cosine_schedule(epoch, warmup_epochs=5, total_epochs=100):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: warmup_cosine_schedule(epoch, 5, args.epochs)
    )
    
    # Pretrain extraction network if specified
    if args.pretrain_extractor:
        print(f"Pretraining extraction network for {args.pretrain_epochs} epochs...")
        pretrain_extraction(
            steg_system, train_loader, val_loader, device,
            num_epochs=args.pretrain_epochs, lr=args.lr * 2
        )
    
    # Training loop
    print(f"Starting main training for {args.epochs} epochs...")
    best_val_psnr_secret = 0
    train_losses = []
    val_metrics_list = []
    
    # Create dynamic loss weight schedule
    # Start with more focus on container quality, gradually shift to secret preservation
    # This helps stabilize early training and prevent secret information loss in later stages
    start_alpha = 0.7  # Initial focus on hiding loss (container quality)
    start_beta = 0.3   # Lower initial weight for extraction loss
    dynamic_alpha = np.linspace(start_alpha, args.alpha, args.epochs)
    dynamic_beta = np.linspace(start_beta, args.beta, args.epochs)
    
    for epoch in range(args.epochs):
        # Get current loss weights based on schedule
        curr_alpha = dynamic_alpha[epoch]
        curr_beta = dynamic_beta[epoch]
        
        print(f"Epoch {epoch+1}/{args.epochs} - Alpha: {curr_alpha:.3f}, Beta: {curr_beta:.3f}")
        
        # Train
        train_loss, train_hiding_loss, train_extraction_loss, train_attention_loss = train_epoch(
            steg_system, train_loader, optimizer, device,
            alpha=curr_alpha, beta=curr_beta, use_high_attention=args.use_high_attention
        )
        
        train_losses.append({
            'epoch': epoch + 1,
            'total_loss': train_loss,
            'hiding_loss': train_hiding_loss,
            'extraction_loss': train_extraction_loss,
            'attention_loss': train_attention_loss
        })
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # Validate
        val_result = validate(
            steg_system, val_loader, device,
            alpha=curr_alpha, beta=curr_beta, use_high_attention=args.use_high_attention
        )
        
        # Unpack validation results - note that validate now returns 8 values including attention_loss
        (val_loss, val_hiding_loss, val_extraction_loss, 
         val_psnr_container, val_psnr_secret, val_ssim_container, val_ssim_secret,
         val_attention_loss) = val_result
        
        # Store validation metrics
        val_metrics_list.append({
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'val_hiding_loss': val_hiding_loss,
            'val_extraction_loss': val_extraction_loss,
            'val_attention_loss': val_attention_loss,
            'val_psnr_container': val_psnr_container,
            'val_psnr_secret': val_psnr_secret,
            'val_ssim_container': val_ssim_container,
            'val_ssim_secret': val_ssim_secret
        })
        
        # Visualize results
        metrics = visualize_results(steg_system, val_loader, device, save_dir, epoch + 1)
        
        # Calculate PSNR and SSIM for both high and low attention embedding
        with torch.no_grad():
            # Get sample images for metrics calculation
            cover_imgs, secret_imgs = select_visualization_samples(val_loader, device)
            
            # Get both high and low attention results
            results = steg_system.compare_attention_methods(cover_imgs, secret_imgs)
            
            # Extract high attention results
            container_high = results["container_high"]
            extracted_high = results["extracted_high"]
            
            # Extract low attention results
            container_low = results["container_low"]
            extracted_low = results["extracted_low"]
            
            # Calculate high attention metrics
            high_psnr_container = 0.0
            high_ssim_container = 0.0
            high_psnr_secret = 0.0
            high_ssim_secret = 0.0
            
            # Calculate low attention metrics
            low_psnr_container = 0.0
            low_ssim_container = 0.0
            low_psnr_secret = 0.0
            low_ssim_secret = 0.0
            
            # Calculate metrics for each image in the batch
            for i in range(cover_imgs.size(0)):
                # Get numpy arrays for metric calculation
                cover_np = cover_imgs[i].cpu().numpy().transpose(1, 2, 0)
                secret_np = secret_imgs[i].cpu().numpy().transpose(1, 2, 0)
                
                # High attention results
                container_high_np = container_high[i].cpu().numpy().transpose(1, 2, 0)
                extracted_high_np = extracted_high[i].cpu().numpy().transpose(1, 2, 0)
                
                # Low attention results
                container_low_np = container_low[i].cpu().numpy().transpose(1, 2, 0)
                extracted_low_np = extracted_low[i].cpu().numpy().transpose(1, 2, 0)
                
                # Calculate high attention metrics
                high_psnr_container += calculate_psnr(cover_np, container_high_np)
                high_ssim_container += calculate_ssim(cover_np, container_high_np)
                high_psnr_secret += calculate_psnr(secret_np, extracted_high_np)
                high_ssim_secret += calculate_ssim(secret_np, extracted_high_np)
                
                # Calculate low attention metrics
                low_psnr_container += calculate_psnr(cover_np, container_low_np)
                low_ssim_container += calculate_ssim(cover_np, container_low_np)
                low_psnr_secret += calculate_psnr(secret_np, extracted_low_np)
                low_ssim_secret += calculate_ssim(secret_np, extracted_low_np)
            
            # Calculate averages
            batch_size = cover_imgs.size(0)
            high_psnr_container /= batch_size
            high_ssim_container /= batch_size
            high_psnr_secret /= batch_size
            high_ssim_secret /= batch_size
            
            low_psnr_container /= batch_size
            low_ssim_container /= batch_size
            low_psnr_secret /= batch_size
            low_ssim_secret /= batch_size
            
            # Print metrics for both embedding methods
            print(f"\nHigh Attention Embedding Metrics:")
            print(f"  Container: PSNR={high_psnr_container:.2f}dB, SSIM={high_ssim_container:.4f}")
            print(f"  Secret: PSNR={high_psnr_secret:.2f}dB, SSIM={high_ssim_secret:.4f}")
            
            print(f"Low Attention Embedding Metrics:")
            print(f"  Container: PSNR={low_psnr_container:.2f}dB, SSIM={low_ssim_container:.4f}")
            print(f"  Secret: PSNR={low_psnr_secret:.2f}dB, SSIM={low_ssim_secret:.4f}\n")
        
        # Print statistics
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f} (Hide: {train_hiding_loss:.4f}, Extract: {train_extraction_loss:.4f}, Attn: {train_attention_loss:.4f}) | "
              f"Val Loss: {val_loss:.4f} (Hide: {val_hiding_loss:.4f}, Extract: {val_extraction_loss:.4f}, Attn: {val_attention_loss:.4f}) | "
              f"Val PSNR - Container: {val_psnr_container:.2f}dB, Secret: {val_psnr_secret:.2f}dB | "
              f"Val SSIM - Container: {val_ssim_container:.4f}, Secret: {val_ssim_secret:.4f}")
        
        # Save checkpoint if best model
        if val_psnr_secret > best_val_psnr_secret:
            best_val_psnr_secret = val_psnr_secret
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': steg_system.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_psnr_secret': val_psnr_secret,
                'val_psnr_container': val_psnr_container,
                'val_ssim_secret': val_ssim_secret,
                'val_ssim_container': val_ssim_container,
                'attention_mode': 'high' if args.use_high_attention else 'low'
            }
            save_checkpoint(checkpoint, f"{save_dir}/best_model.pth")
            print(f"Saved best model with Secret PSNR: {val_psnr_secret:.2f}dB")
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': steg_system.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_psnr_secret': val_psnr_secret,
                'val_psnr_container': val_psnr_container,
                'val_ssim_secret': val_ssim_secret,
                'val_ssim_container': val_ssim_container,
                'attention_mode': 'high' if args.use_high_attention else 'low'
            }
            save_checkpoint(checkpoint, f"{save_dir}/epoch_{epoch+1}.pth")
    
    # Plot training curves
    epochs = [entry['epoch'] for entry in train_losses]
    
    # Training losses
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, [entry['total_loss'] for entry in train_losses], label='Total Loss')
    plt.plot(epochs, [entry['hiding_loss'] for entry in train_losses], label='Hiding Loss')
    plt.plot(epochs, [entry['extraction_loss'] for entry in train_losses], label='Extraction Loss')
    plt.plot(epochs, [entry['attention_loss'] for entry in train_losses], label='Attention Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/training_losses.png", dpi=300, bbox_inches='tight')
    
    # Validation metrics
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, [entry['val_psnr_container'] for entry in val_metrics_list], label='Container PSNR')
    plt.plot(epochs, [entry['val_psnr_secret'] for entry in val_metrics_list], label='Secret PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('Validation PSNR')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, [entry['val_ssim_container'] for entry in val_metrics_list], label='Container SSIM')
    plt.plot(epochs, [entry['val_ssim_secret'] for entry in val_metrics_list], label='Secret SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Validation SSIM')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/validation_metrics.png", dpi=300, bbox_inches='tight')
    
    print("Training complete. Final results saved to:", save_dir)

if __name__ == "__main__":
    main()

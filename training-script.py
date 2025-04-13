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
    Combined loss function that balances L1 and SSIM loss
    for better perceptual quality
    """
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        # L1 loss for overall structure
        l1 = self.l1_loss(pred, target)
        
        # MSE loss for optimization stability
        mse = self.mse_loss(pred, target)
        
        # Combined loss - mostly L1 with some MSE
        return self.alpha * l1 + (1 - self.alpha) * mse

def train_epoch(steg_system, dataloader, optimizer, device, 
                alpha=0.5, beta=0.5, use_high_attention=True):
    """
    Train for one epoch using the enhanced steganography system
    
    Args:
        steg_system: EnhancedSteganographySystem model
        dataloader: DataLoader for the training set
        optimizer: Optimizer for the system
        device: Device to use (cuda/cpu)
        alpha: Weight for cover-container loss
        beta: Weight for secret-extracted loss
        use_high_attention: Whether to use high attention regions (True) or low attention regions (False)
    
    Returns:
        average_loss: Average loss for the epoch
    """
    steg_system.train()
    
    total_loss = 0.0
    total_hiding_loss = 0.0
    total_extraction_loss = 0.0
    
    criterion = CombinedLoss(alpha=0.7)
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for i, (cover_imgs, secret_imgs) in enumerate(progress_bar):
        # Cover and secret images are already different (from the dataset)
        cover_imgs = cover_imgs.to(device)
        secret_imgs = secret_imgs.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass - using the enhanced system
        container_imgs, cover_attention, secret_attention, embedding_map = steg_system.forward_hide(
            cover_imgs, secret_imgs, use_high_attention=use_high_attention
        )
        extracted_secrets = steg_system.forward_extract(container_imgs)
        
        # Calculate losses
        hiding_loss = criterion(container_imgs, cover_imgs)  # Container should look like cover
        extraction_loss = criterion(extracted_secrets, secret_imgs)  # Extracted should match secret
        
        # Combined loss
        loss = alpha * hiding_loss + beta * extraction_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update stats
        total_loss += loss.item()
        total_hiding_loss += hiding_loss.item()
        total_extraction_loss += extraction_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'hiding_loss': f'{hiding_loss.item():.4f}',
            'extract_loss': f'{extraction_loss.item():.4f}'
        })
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_hiding_loss = total_hiding_loss / len(dataloader)
    avg_extraction_loss = total_extraction_loss / len(dataloader)
    
    return avg_loss, avg_hiding_loss, avg_extraction_loss

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
        # Extract single sample tensors
        cover_np = cover_imgs[i].cpu().numpy().transpose(1, 2, 0)
        secret_np = secret_imgs[i].cpu().numpy().transpose(1, 2, 0)
        cover_atn_np = cover_attention[i].cpu().numpy().squeeze()
        secret_atn_np = secret_attention[i].cpu().numpy().squeeze()
        embed_map_high_np = embedding_map_high[i].cpu().numpy().squeeze()
        embed_map_low_np = embedding_map_low[i].cpu().numpy().squeeze()
        container_high_np = container_high[i].cpu().numpy().transpose(1, 2, 0)
        container_low_np = container_low[i].cpu().numpy().transpose(1, 2, 0)
        extracted_high_np = extracted_high[i].cpu().numpy().transpose(1, 2, 0)
        extracted_low_np = extracted_low[i].cpu().numpy().transpose(1, 2, 0)
        
        # Display images
        axes[i, 0].imshow(np.clip(cover_np, 0, 1))
        axes[i, 0].set_title('Cover Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(np.clip(secret_np, 0, 1))
        axes[i, 1].set_title('Secret Image')
        axes[i, 1].axis('off')
        
        # Display attention heatmaps
        axes[i, 2].imshow(cover_atn_np, cmap='hot')
        axes[i, 2].set_title('Cover Attention')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(secret_atn_np, cmap='hot')
        axes[i, 3].set_title('Secret Attention')
        axes[i, 3].axis('off')
        
        # Display embedding maps
        axes[i, 4].imshow(embed_map_high_np, cmap='viridis')
        axes[i, 4].set_title('High Attn Embedding')
        axes[i, 4].axis('off')
        
        axes[i, 5].imshow(embed_map_low_np, cmap='viridis')
        axes[i, 5].set_title('Low Attn Embedding')
        axes[i, 5].axis('off')
        
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
            axes[i, 6].axis('off')
            
            axes[i, 7].imshow(np.clip(extracted_np, 0, 1))
            axes[i, 7].set_title(f'Extracted (High)\nPSNR: {psnr_high_secret:.2f}dB\nSSIM: {ssim_high_secret:.4f}')
            axes[i, 7].axis('off')
        else:
            container_np = container_low_np
            extracted_np = extracted_low_np
            
            axes[i, 6].imshow(np.clip(container_np, 0, 1))
            axes[i, 6].set_title(f'Container (Low)\nPSNR: {psnr_low_container:.2f}dB\nSSIM: {ssim_low_container:.4f}')
            axes[i, 6].axis('off')
            
            axes[i, 7].imshow(np.clip(extracted_np, 0, 1))
            axes[i, 7].set_title(f'Extracted (Low)\nPSNR: {psnr_low_secret:.2f}dB\nSSIM: {ssim_low_secret:.4f}')
            axes[i, 7].axis('off')
    
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
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train with specified attention mode
        train_loss, train_hiding_loss, train_extraction_loss = train_epoch(
            steg_system, train_loader, optimizer, device, 
            alpha=args.alpha, beta=args.beta, 
            use_high_attention=args.use_high_attention
        )
        
        # If train_both flag is set, also train with the opposite attention mode
        if args.train_both:
            print(f"Training with {'low' if args.use_high_attention else 'high'} attention regions...")
            opposite_train_loss, opposite_train_hiding_loss, opposite_train_extraction_loss = train_epoch(
                steg_system, train_loader, optimizer, device, 
                alpha=args.alpha, beta=args.beta, 
                use_high_attention=not args.use_high_attention
            )
        
        # Validate with specified attention mode
        val_metrics = validate(
            steg_system, val_loader, device, 
            alpha=args.alpha, beta=args.beta, 
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
                alpha=args.alpha, beta=args.beta, 
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
            print(f"Train Loss: {opposite_train_loss:.4f}, Hiding Loss: {opposite_train_hiding_loss:.4f}, Extraction Loss: {opposite_train_extraction_loss:.4f}")
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
                    'train_loss': opposite_train_loss,
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
            print("\n⭐ High attention embedding performed better overall")
        else:
            print("\n⭐ Low attention embedding performed better overall")
        
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

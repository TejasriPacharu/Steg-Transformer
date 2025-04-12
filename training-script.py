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
import torch

# If they're in the same file, you can import them directly
from swin_model import HidingNetwork, ExtractionNetwork, calculate_psnr, calculate_ssim, calculate_mse


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
   
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_paths = []
        
        if split == 'train':
            # Process training data
            train_dir = os.path.join(root_dir, 'train')
            if not os.path.exists(train_dir):
                print(f"Train directory '{train_dir}' does not exist.")
                return

            for class_folder in os.listdir(train_dir):
                class_path = os.path.join(train_dir, class_folder)
                images_path = os.path.join(class_path, 'images')
            
                if os.path.isdir(images_path):
                    for img_file in os.listdir(images_path):
                        if img_file.endswith('.JPEG'):
                            self.img_paths.append(os.path.join(images_path, img_file))
        
        elif split == 'val':
            # Process validation data
            val_dir = os.path.join(root_dir, 'val')
            for class_folder in os.listdir(val_dir):
                class_path = os.path.join(val_dir, class_folder, 'images')
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.endswith('.JPEG'):
                            self.img_paths.append(os.path.join(class_path, img_file))
        
        print(f"Loaded {len(self.img_paths)} images for {split}")
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

def train_epoch(hiding_net, extraction_net, dataloader, criterion_mse, optimizer_h, optimizer_e, device, alpha=0.5, beta=0.5):
    """
    Train for one epoch
    
    Args:
        hiding_net: Hiding network model
        extraction_net: Extraction network model
        dataloader: DataLoader for the training set
        criterion_mse: MSE loss function
        optimizer_h: Optimizer for hiding network
        optimizer_e: Optimizer for extraction network
        device: Device to use (cuda/cpu)
        alpha: Weight for cover-container loss
        beta: Weight for secret-extracted loss
    
    Returns:
        average_loss: Average loss for the epoch
    """
    hiding_net.train()
    extraction_net.train()
    
    total_loss = 0.0
    total_hiding_loss = 0.0
    total_extraction_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for i, (cover_imgs, secret_imgs) in enumerate(progress_bar):
        # Cover and secret images are already different (from the dataset)
        cover_imgs = cover_imgs.to(device)
        secret_imgs = secret_imgs.to(device)
        
        # Zero the parameter gradients
        optimizer_h.zero_grad()
        optimizer_e.zero_grad()
        
        # Forward pass
        container_imgs = hiding_net(cover_imgs, secret_imgs)
        extracted_secrets = extraction_net(container_imgs)
        
        # Calculate losses
        hiding_loss = criterion_mse(container_imgs, cover_imgs)  # Container should look like cover
        extraction_loss = criterion_mse(extracted_secrets, secret_imgs)  # Extracted should match secret
        
        # Combined loss
        loss = alpha * hiding_loss + beta * extraction_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer_h.step()
        optimizer_e.step()
        
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

def validate(hiding_net, extraction_net, dataloader, criterion_mse, device, alpha=0.5, beta=0.5):
    """
    Validate the model
    
    Args:
        hiding_net: Hiding network model
        extraction_net: Extraction network model
        dataloader: DataLoader for the validation set
        criterion_mse: MSE loss function
        device: Device to use (cuda/cpu)
        alpha: Weight for cover-container loss
        beta: Weight for secret-extracted loss
    
    Returns:
        average_loss: Average loss for the validation set
        psnr_container: Average PSNR between cover and container
        psnr_secret: Average PSNR between secret and extracted
    """
    hiding_net.eval()
    extraction_net.eval()
    
    total_loss = 0.0
    total_hiding_loss = 0.0
    total_extraction_loss = 0.0
    total_psnr_container = 0.0
    total_psnr_secret = 0.0
    total_ssim_container = 0.0
    total_ssim_secret = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validating')
        
        for i, (cover_imgs, secret_imgs) in enumerate(progress_bar):
            # Cover and secret images are already different (from the dataset)
            cover_imgs = cover_imgs.to(device)
            secret_imgs = secret_imgs.to(device)
            
            # Forward pass
            container_imgs = hiding_net(cover_imgs, secret_imgs)
            extracted_secrets = extraction_net(container_imgs)
            
            # Calculate losses
            hiding_loss = criterion_mse(container_imgs, cover_imgs)
            extraction_loss = criterion_mse(extracted_secrets, secret_imgs)
            
            # Combined loss
            loss = alpha * hiding_loss + beta * extraction_loss
            
            # Calculate metrics
            for j in range(cover_imgs.size(0)):
                # Convert to numpy for PSNR and SSIM calculation
                cover_np = cover_imgs[j].cpu().numpy().transpose(1, 2, 0) * 255.0
                container_np = container_imgs[j].cpu().numpy().transpose(1, 2, 0) * 255.0
                secret_np = secret_imgs[j].cpu().numpy().transpose(1, 2, 0) * 255.0
                extracted_np = extracted_secrets[j].cpu().numpy().transpose(1, 2, 0) * 255.0
                
                # Calculate PSNR
                psnr_container = calculate_psnr(cover_np.astype(np.uint8), container_np.astype(np.uint8))
                psnr_secret = calculate_psnr(secret_np.astype(np.uint8), extracted_np.astype(np.uint8))
                
                # Calculate SSIM
                ssim_container = calculate_ssim(cover_np.astype(np.uint8), container_np.astype(np.uint8))
                ssim_secret = calculate_ssim(secret_np.astype(np.uint8), extracted_np.astype(np.uint8))
                
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
                'hiding_loss': f'{hiding_loss.item():.4f}',
                'extract_loss': f'{extraction_loss.item():.4f}'
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

def visualize_results(hiding_net, extraction_net, dataloader, device, save_dir, epoch):
    """
    Visualize the results on a few sample images
    
    Args:
        hiding_net: Hiding network model
        extraction_net: Extraction network model
        dataloader: DataLoader for the validation set
        device: Device to use (cuda/cpu)
        save_dir: Directory to save the visualizations
        epoch: Current epoch number
    """
    hiding_net.eval()
    extraction_net.eval()
    
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
        # Forward pass
        container_imgs = hiding_net(cover_imgs, secret_imgs)
        extracted_secrets = extraction_net(container_imgs)
    
    # Create a figure with n_samples rows (one for each sample) and 4 columns (cover, secret, container, extracted)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    # Handle case where n_samples is 1 (axes would be 1D)
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Convert tensors to numpy for visualization
        cover_np = cover_imgs[i].cpu().numpy().transpose(1, 2, 0)
        secret_np = secret_imgs[i].cpu().numpy().transpose(1, 2, 0)
        container_np = container_imgs[i].cpu().numpy().transpose(1, 2, 0)
        extracted_np = extracted_secrets[i].cpu().numpy().transpose(1, 2, 0)
        
        # Calculate PSNR
        psnr_container = calculate_psnr((cover_np * 255).astype(np.uint8), (container_np * 255).astype(np.uint8))
        psnr_secret = calculate_psnr((secret_np * 255).astype(np.uint8), (extracted_np * 255).astype(np.uint8))
        
        # Calculate SSIM
        ssim_container = calculate_ssim((cover_np * 255).astype(np.uint8), (container_np * 255).astype(np.uint8))
        ssim_secret = calculate_ssim((secret_np * 255).astype(np.uint8), (extracted_np * 255).astype(np.uint8))
        
        # Plot images
        axes[i, 0].imshow(np.clip(cover_np, 0, 1))
        axes[i, 0].set_title('Cover Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(np.clip(secret_np, 0, 1))
        axes[i, 1].set_title('Secret Image')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(np.clip(container_np, 0, 1))
        axes[i, 2].set_title(f'Container\nPSNR: {psnr_container:.2f}dB\nSSIM: {ssim_container:.4f}')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(np.clip(extracted_np, 0, 1))
        axes[i, 3].set_title(f'Extracted Secret\nPSNR: {psnr_secret:.2f}dB\nSSIM: {ssim_secret:.4f}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'results_epoch_{epoch}.png'))
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
    parser = argparse.ArgumentParser(description='Train SWinT model on Tiny-ImageNet')
    parser.add_argument('--data_dir', type=str, default='./', help='Path to tiny-imagenet-200 directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=144, help='Image size')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for hiding loss')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight for extraction loss')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--vis_dir', type=str, default='./visualizations', help='Directory to save visualizations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for SWinT')
    parser.add_argument('--embed_dim', type=int, default=96, help='Embedding dimension')
    parser.add_argument('--num_heads', type=list, default=[6, 6, 6, 6], help='Number of heads in each layer')
    parser.add_argument('--depths', type=list, default=[2, 2, 2, 2], help='Depth of each layer')
    
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
    
    # Create models
    hiding_net = HidingNetwork(
        img_size=args.img_size,
        window_size=args.window_size,
        embed_dim=args.embed_dim,
        depths=args.depths,
        num_heads=args.num_heads
    ).to(device)
    
    extraction_net = ExtractionNetwork(
        img_size=args.img_size,
        window_size=args.window_size,
        embed_dim=args.embed_dim,
        depths=args.depths,
        num_heads=args.num_heads
    ).to(device)
    
    # Define loss function
    criterion_mse = nn.MSELoss()
    
    # Define optimizers
    optimizer_h = optim.Adam(hiding_net.parameters(), lr=args.lr)
    optimizer_e = optim.Adam(extraction_net.parameters(), lr=args.lr)
    
    # Learning rate schedulers
    scheduler_h = optim.lr_scheduler.StepLR(optimizer_h, step_size=10, gamma=0.5)
    scheduler_e = optim.lr_scheduler.StepLR(optimizer_e, step_size=10, gamma=0.5)
    
    # Initialize variables
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            
            hiding_net.load_state_dict(checkpoint['hiding_state_dict'])
            extraction_net.load_state_dict(checkpoint['extraction_state_dict'])
            
            optimizer_h.load_state_dict(checkpoint['optimizer_h_state_dict'])
            optimizer_e.load_state_dict(checkpoint['optimizer_e_state_dict'])
            
            scheduler_h.load_state_dict(checkpoint['scheduler_h_state_dict'])
            scheduler_e.load_state_dict(checkpoint['scheduler_e_state_dict'])
            
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_hiding_loss, train_extraction_loss = train_epoch(
            hiding_net, extraction_net, train_loader, criterion_mse, 
            optimizer_h, optimizer_e, device, args.alpha, args.beta
        )
        
        # Validate
        val_metrics = validate(
            hiding_net, extraction_net, val_loader, criterion_mse, 
            device, args.alpha, args.beta
        )
        
        val_loss, val_hiding_loss, val_extraction_loss, val_psnr_container, val_psnr_secret, val_ssim_container, val_ssim_secret = val_metrics
        
        # Visualize results
        visualize_results(hiding_net, extraction_net, val_loader, device, args.vis_dir, epoch)
        
        # Step the schedulers
        scheduler_h.step()
        scheduler_e.step()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Hiding Loss: {train_hiding_loss:.4f}, Extraction Loss: {train_extraction_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Hiding Loss: {val_hiding_loss:.4f}, Extraction Loss: {val_extraction_loss:.4f}")
        print(f"PSNR Container: {val_psnr_container:.2f}dB, PSNR Secret: {val_psnr_secret:.2f}dB")
        print(f"SSIM Container: {val_ssim_container:.4f}, SSIM Secret: {val_ssim_secret:.4f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'hiding_state_dict': hiding_net.state_dict(),
            'extraction_state_dict': extraction_net.state_dict(),
            'optimizer_h_state_dict': optimizer_h.state_dict(),
            'optimizer_e_state_dict': optimizer_e.state_dict(),
            'scheduler_h_state_dict': scheduler_h.state_dict(),
            'scheduler_e_state_dict': scheduler_e.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr_container': val_psnr_container,
            'val_psnr_secret': val_psnr_secret,
            'val_ssim_container': val_ssim_container,
            'val_ssim_secret': val_ssim_secret,
        }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'hiding_state_dict': hiding_net.state_dict(),
                'extraction_state_dict': extraction_net.state_dict(),
                'optimizer_h_state_dict': optimizer_h.state_dict(),
                'optimizer_e_state_dict': optimizer_e.state_dict(),
                'scheduler_h_state_dict': scheduler_h.state_dict(),
                'scheduler_e_state_dict': scheduler_e.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_psnr_container': val_psnr_container,
                'val_psnr_secret': val_psnr_secret,
                'val_ssim_container': val_ssim_container,
                'val_ssim_secret': val_ssim_secret,
            }, os.path.join(args.save_dir, 'best_model.pth'))
    
    print("Training completed!")

if __name__ == "__main__":
    main()

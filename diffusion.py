import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from tqdm import tqdm
import os
import random

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn.functional as F
# For plotting
import matplotlib.pyplot as plt
# Import diffusion model utilities
from diffusers import DDPMScheduler, UNet2DModel
import scipy.ndimage

# ============================
# Helper Functions
# ============================

def generate_random_mask(H, W, target_percent=0.2):
    """
    Generate a random brush stroke mask that covers approximately target_percent of the image.

    Returns:
        mask_np (numpy.ndarray): A 2D numpy array representing the mask.
    """
    from PIL import Image, ImageDraw

    mask = Image.new('L', (W, H), color=255)  # White image (255)
    draw = ImageDraw.Draw(mask)

    # Keep adding strokes until the target percent of the image is masked
    while True:
        num_strokes = random.randint(1, 4)
        for _ in range(num_strokes):
            # Random starting point
            start_x = random.randint(0, W)
            start_y = random.randint(0, H)

            # Random brush properties
            brush_width = random.randint(3, 7)  # Adjusted for smaller images
            num_vertices = random.randint(2, 5)
            angles = np.random.uniform(0, 2 * np.pi, size=(num_vertices,))
            lengths = np.random.uniform(5, 15, size=(num_vertices,))  # Adjusted for smaller images

            # Generate stroke points
            points = [(start_x, start_y)]
            for angle, length in zip(angles, lengths):
                dx = int(length * np.cos(angle))
                dy = int(length * np.sin(angle))
                new_x = np.clip(points[-1][0] + dx, 0, W - 1)
                new_y = np.clip(points[-1][1] + dy, 0, H - 1)
                points.append((new_x, new_y))

            # Draw the stroke
            draw.line(points, fill=0, width=brush_width, joint='curve')

            # Optional: add circles at points to simulate brush pressure
            for point in points:
                radius = brush_width // 2
                bbox = [point[0] - radius, point[1] - radius,
                        point[0] + radius, point[1] + radius]
                draw.ellipse(bbox, fill=0)

        # Convert mask to numpy array and calculate masked percentage
        mask_np = np.array(mask) / 255.0  # 1 for known regions (white), 0 for masked regions (black)
        percent_masked = 1.0 - np.mean(mask_np)
        if percent_masked >= target_percent:
            break

    return mask_np

def compute_covariance_matrices(E_x, E_y, mask, epsilon=1e-5):
    """
    Estimate the covariance matrix based on local gradients.

    Returns:
        covariance_matrices (torch.Tensor): Covariance matrices for each pixel.
    """
    H, W = mask.shape
    window_size = 3  # Adjusted window size for smaller images
    pad = window_size // 2

    # Structure tensor components
    E_x2 = E_x ** 2
    E_y2 = E_y ** 2
    E_xy = E_x * E_y

    # Sum over local neighborhood
    kernel = torch.ones((1, 1, window_size, window_size)).to(E_x.device)

    J11 = F.conv2d(E_x2.unsqueeze(0).unsqueeze(0), kernel, padding=pad).squeeze()
    J22 = F.conv2d(E_y2.unsqueeze(0).unsqueeze(0), kernel, padding=pad).squeeze()
    J12 = F.conv2d(E_xy.unsqueeze(0).unsqueeze(0), kernel, padding=pad).squeeze()

    # Regularization and inversion
    epsilon_matrix = epsilon * torch.ones_like(J11).to(E_x.device)
    det = J11 * J22 - J12 ** 2 + epsilon_matrix
    inv_J11 = J22 / det
    inv_J22 = J11 / det
    inv_J12 = -J12 / det

    # Covariance matrices
    covariance_matrices = torch.stack([inv_J11, inv_J12, inv_J12, inv_J22], dim=-1).view(H, W, 2, 2)

    # Apply mask
    covariance_matrices = covariance_matrices * mask.unsqueeze(-1).unsqueeze(-1)

    return covariance_matrices  # [H, W, 2, 2]

def compute_amplitude_map(mask, beta=0.1):
    """
    Compute amplitude based on distance to known regions.

    Returns:
        amplitude_map (torch.Tensor): Amplitude map.
    """
    known_mask = mask  # Since mask has 1s in known regions
    known_mask_np = known_mask.cpu().numpy()
    distance_map = scipy.ndimage.distance_transform_edt(1 - known_mask_np)
    amplitude_map = torch.from_numpy(distance_map).float().to(mask.device)
    amplitude_map = torch.exp(-beta * amplitude_map)
    return amplitude_map  # [H, W]

def compute_gaussian_splat_map(mask, covariance_matrices, amplitude_map, scales=[1, 2, 4]):
    """
    Generate the Gaussian splat map with multi-scale integration.

    Returns:
        S_norm (torch.Tensor): Normalized Gaussian splat map.
    """
    H, W = mask.shape
    S_total = torch.zeros(H, W).to(mask.device)

    for scale in scales:
        scaled_cov_matrices = covariance_matrices / scale  # Adjust scaling
        S = compute_gaussian_splat_map_single_scale(mask, scaled_cov_matrices, amplitude_map)
        S_total += S

    # Normalize
    S_min = S_total.min()
    S_max = S_total.max()
    S_norm = (S_total - S_min) / (S_max - S_min + 1e-8)
    return S_norm  # [H, W]

def compute_gaussian_splat_map_single_scale(mask, covariance_matrices, amplitude_map):
    """
    Compute Gaussian splat map at a single scale.

    Returns:
        S (torch.Tensor): Gaussian splat map.
    """
    H, W = mask.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=mask.device), torch.arange(W, device=mask.device), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).float()  # [H, W, 2]

    indices = torch.nonzero(mask == 0).to(mask.device)  # Only masked regions
    if indices.size(0) == 0:
        return torch.zeros(H, W).to(mask.device)

    # Extract positions and covariance matrices
    positions = indices[:, [1, 0]].float()  # [N, 2] (x, y)
    amplitudes = amplitude_map[indices[:, 0], indices[:, 1]]  # [N]
    cov_inv = covariance_matrices[indices[:, 0], indices[:, 1]]  # [N, 2, 2]

    # Compute differences
    grid_flat = grid.view(-1, 2)  # [H*W, 2]
    diff = grid_flat.unsqueeze(0) - positions.unsqueeze(1)  # [N, H*W, 2]

    # Compute exponentials
    exponent = -0.5 * torch.sum((diff @ cov_inv) * diff, dim=2)  # [N, H*W]

    G = torch.exp(exponent)  # [N, H*W]

    # Weighted sum
    S = torch.sum(amplitudes.unsqueeze(1) * G, dim=0)  # [H*W]
    S = S.view(H, W)

    return S  # [H, W]


# ============================
# Main Training Function
# ============================

def main(rank, world_size):
    print(f"Rank {rank} starting training.")
    try:
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        # Initialize the process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        print(f"Rank {rank} initialized process group.")
        
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
        print(f"Rank {rank} set to device {device}.")

        # Load CIFAR-10 Dataset
        print(f"Rank {rank} loading CIFAR-10 dataset.")

        # Define transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Normalize to [-1, 1]
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

        train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=4, pin_memory=True)

        # Define Sobel kernels once
        sobel_kernel_x = torch.tensor([[[[-1, 0, 1],
                                         [-2, 0, 2],
                                         [-1, 0, 1]]]], dtype=torch.float32).to(device)
        sobel_kernel_y = torch.tensor([[[[-1, -2, -1],
                                         [0, 0, 0],
                                         [1, 2, 1]]]], dtype=torch.float32).to(device)

        # Initialize models and wrap with DDP
        # Adjusted input channels to 7 (3 for image + 1 for mask + 1 for S_norm + 1 for edge map + 1 for attention map)
        # Simplified model architecture for faster training
        model = UNet2DModel(
            sample_size=32,  # CIFAR-10 images are 32x32
            in_channels=7,   # Number of input channels
            out_channels=3,  # RGB channels
            layers_per_block=1,  # Reduced from 2 to 1
            block_out_channels=(64, 128, 256),  # Reduced number of channels
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        ).to(device)

        # Convert BatchNorm layers to SyncBatchNorm if any
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Wrap model with DDP
        model = DDP(model, device_ids=[rank])

        # Define the scheduler (the diffusion process)
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

        # Ensure alphas_cumprod are on the correct device and type
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device).float()

        # Set number of inference steps
        num_inference_steps = 50  # You can adjust this value as needed

        # Define optimizer with reduced learning rate
        optimizer = optim.AdamW(model.parameters(), lr=2e-4)

        # Load VGG19 for perceptual and style losses
        vgg_model = torchvision.models.vgg19(pretrained=True).features.to(device).eval()
        for param in vgg_model.parameters():
            param.requires_grad = False

        # Loss functions
        criterion_l1 = nn.L1Loss().to(device)
        criterion_mse = nn.MSELoss().to(device)

        # Define convolution for attention map
        attention_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(device)
        nn.init.xavier_uniform_(attention_conv.weight)

        # Create plots directory if it doesn't exist (only on rank 0)
        if rank == 0:
            os.makedirs('./plots', exist_ok=True)

        # Training loop
        num_epochs = 200  # Adjust the number of epochs as needed
        lambda_noise = 1.0
        lambda_rec = 5.0
        lambda_perc = 0.5
        lambda_style = 1.0  # Reduced from 100 to 1
        lambda_tv = 0.05

        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            if rank == 0:
                pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
            else:
                pbar = train_loader

            for data in pbar:
                real_images, _ = data
                real_images = real_images.to(device, non_blocking=True)
                batch_size, C, H, W = real_images.size()

                # Generate masks
                masks = []
                for _ in range(batch_size):
                    mask = generate_random_mask(H, W)
                    masks.append(mask)
                masks = np.stack(masks, axis=0)
                masks = torch.from_numpy(masks).unsqueeze(1).float().to(device, non_blocking=True)  # [B, 1, H, W]

                # Incomplete images (masked images)
                incomplete_images = real_images * masks  # Known regions remain, masked regions are zero

                # Compute gradients
                gray_images = torch.mean(real_images, dim=1, keepdim=True)
                E_x = F.conv2d(gray_images, sobel_kernel_x, padding=1)
                E_y = F.conv2d(gray_images, sobel_kernel_y, padding=1)

                # Compute edge maps
                edge_maps = torch.sqrt(E_x ** 2 + E_y ** 2)

                # Compute covariance matrices, amplitude maps, and Gaussian splat maps
                covariance_matrices_list = []
                amplitude_maps_list = []
                S_list = []
                for i in range(batch_size):
                    cov_matrices = compute_covariance_matrices(E_x[i, 0], E_y[i, 0], masks[i, 0])
                    covariance_matrices_list.append(cov_matrices)

                    amplitude_map = compute_amplitude_map(masks[i, 0])
                    amplitude_maps_list.append(amplitude_map)

                    S_norm = compute_gaussian_splat_map(masks[i, 0], cov_matrices, amplitude_map)
                    S_list.append(S_norm)

                S_norm = torch.stack(S_list, dim=0).unsqueeze(1)  # [B, 1, H, W]

                # Compute attention map from S_norm
                attention_map = torch.sigmoid(attention_conv(S_norm))

                # Create noisy complete images
                noise = torch.randn_like(real_images)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

                alpha_t = scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

                noisy_real_images = sqrt_alpha_t * real_images + sqrt_one_minus_alpha_t * noise

                # Prepare model inputs with full noisy images
                model_inputs = torch.cat([noisy_real_images, masks, S_norm, edge_maps, attention_map], dim=1)  # [B, 7, H, W]

                # Forward pass
                optimizer.zero_grad()
                model_output = model(model_inputs, timesteps).sample  # [B, 3, H, W]

                # Compute the predicted noise residual
                epsilon_theta = model_output

                # Noise prediction loss computed over masked regions
                masks_expanded_C = (1 - masks)  # [B, 1, H, W]
                noise_diff = epsilon_theta - noise
                noise_loss = criterion_mse(epsilon_theta * masks_expanded_C, noise * masks_expanded_C)

                # Reconstructed image estimate
                x0_pred = (noisy_real_images - sqrt_one_minus_alpha_t * epsilon_theta) / sqrt_alpha_t

                # Reconstruction loss over masked regions
                rec_loss = criterion_l1(x0_pred * masks_expanded_C, real_images * masks_expanded_C)

                # Perceptual loss over masked regions
                def compute_perceptual_loss(x, y, mask):
                    loss = 0.0
                    layers = [3, 8, 15]  # Fewer layers for faster computation
                    x_features = x
                    y_features = y
                    for i, layer in enumerate(vgg_model):
                        x_features = layer(x_features)
                        y_features = layer(y_features)
                        if i in layers:
                            # Resize mask to match feature map size
                            current_mask = F.interpolate(mask, size=x_features.shape[2:], mode='nearest')  # [B, 1, H', W']
                            # Focus on masked regions
                            masked_x = x_features * (1 - current_mask)
                            masked_y = y_features * (1 - current_mask)
                            loss += criterion_l1(masked_x, masked_y)
                        if i >= max(layers):
                            break
                    return loss

                perc_loss = compute_perceptual_loss(x0_pred, real_images, masks)

                # Style loss over masked regions
                def compute_gram_matrix(feat, mask):
                    B, C, H, W = feat.size()
                    # Resize mask to match feature map size
                    current_mask = F.interpolate(mask, size=(H, W), mode='nearest')  # [B, 1, H, W]
                    # Focus on masked regions
                    masked_feat = feat * (1 - current_mask)
                    # Flatten spatial dimensions
                    masked_feat_flat = masked_feat.view(B, C, -1)  # [B, C, H*W]
                    # Compute gram matrix
                    gram = torch.bmm(masked_feat_flat, masked_feat_flat.transpose(1, 2)) / (C * H * W)
                    return gram

                def compute_style_loss(x, y, mask):
                    loss = 0.0
                    layers = [3, 8, 15]
                    x_features = x
                    y_features = y
                    for i, layer in enumerate(vgg_model):
                        x_features = layer(x_features)
                        y_features = layer(y_features)
                        if i in layers:
                            x_gram = compute_gram_matrix(x_features, mask)
                            y_gram = compute_gram_matrix(y_features, mask)
                            loss += criterion_l1(x_gram, y_gram)
                        if i >= max(layers):
                            break
                    return loss

                style_loss = compute_style_loss(x0_pred, real_images, masks)

                # Total variation loss over the reconstructed image (masked regions)
                dx = x0_pred[:, :, :, 1:] - x0_pred[:, :, :, :-1]
                dy = x0_pred[:, :, 1:, :] - x0_pred[:, :, :-1, :]
                mask_dx = (1 - masks[:, :, :, 1:]) * (1 - masks[:, :, :, :-1])
                mask_dy = (1 - masks[:, :, 1:, :]) * (1 - masks[:, :, :-1, :])

                tv_loss = torch.mean(torch.abs(dx * mask_dx)) + torch.mean(torch.abs(dy * mask_dy))

                # Total loss
                total_loss = lambda_noise * noise_loss + \
                             lambda_rec * rec_loss + \
                             lambda_perc * perc_loss + \
                             lambda_style * style_loss + \
                             lambda_tv * tv_loss

                total_loss.backward()
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Update progress bar only on rank 0
                if rank == 0:
                    pbar.set_postfix({
                        'Total Loss': f'{total_loss.item():.4f}',
                        'Noise Loss': f'{noise_loss.item():.4f}',
                        'Rec Loss': f'{rec_loss.item():.4f}',
                        'Perc Loss': f'{perc_loss.item():.4f}',
                        'Style Loss': f'{style_loss.item():.4f}',
                        'TV Loss': f'{tv_loss.item():.4f}',
                    })

            # At the end of each epoch, only rank 0 outputs statistics, saves plots, and saves the model
            if rank == 0:
                # Logging
                with open('log.txt', 'a') as f:
                    f.write(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss.item():.4f}\n")

                # Save images every 10 epochs
                if (epoch + 1) % 10 == 0:
                    # Generate images using reverse diffusion
                    n_images = 4  # Number of images to generate

                    # Get samples for visualization
                    real_images_sample = real_images[:n_images]
                    masks_sample = masks[:n_images]
                    incomplete_images_sample = incomplete_images[:n_images]
                    S_norm_sample = S_norm[:n_images]
                    edge_maps_sample = edge_maps[:n_images]
                    attention_map_sample = attention_map[:n_images]

                    # Initialize noisy images
                    generated_images = torch.randn_like(real_images_sample).to(device)

                    # Start reverse diffusion process
                    scheduler.set_timesteps(num_inference_steps)

                    # Retrieve alphas and betas for the timesteps
                    alphas = scheduler.alphas_cumprod.to(device)
                    sqrt_alphas = torch.sqrt(alphas)
                    sqrt_one_minus_alphas = torch.sqrt(1 - alphas)

                    for t in scheduler.timesteps:
                        timesteps_sample = torch.tensor([t] * n_images, device=device).long()

                        # Prepare model inputs
                        with torch.no_grad():
                            model_inputs = torch.cat([generated_images, masks_sample, S_norm_sample, edge_maps_sample, attention_map_sample], dim=1)
                            # Predict noise residual
                            noise_pred = model(model_inputs, timesteps_sample).sample

                        # Compute previous image
                        generated_images = scheduler.step(noise_pred, t, generated_images).prev_sample

                        # Enforce known pixels
                        generated_images = generated_images * (1 - masks_sample) + real_images_sample * masks_sample

                    # Save generated images
                    for i in range(n_images):
                        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                        # Original image
                        original = real_images_sample[i].cpu().permute(1, 2, 0).numpy()
                        original = (original * 0.5 + 0.5).clip(0, 1)
                        axs[0].imshow(original)
                        axs[0].set_title('Original Image')
                        axs[0].axis('off')

                        # Masked image
                        masked = incomplete_images_sample[i].cpu().permute(1, 2, 0).numpy()
                        masked = (masked * 0.5 + 0.5).clip(0, 1)
                        axs[1].imshow(masked)
                        axs[1].set_title('Masked Image')
                        axs[1].axis('off')

                        # Reconstructed image
                        reconstructed = generated_images[i].cpu().permute(1, 2, 0).numpy()
                        reconstructed = (reconstructed * 0.5 + 0.5).clip(0, 1)
                        axs[2].imshow(reconstructed)
                        axs[2].set_title('Reconstructed Image')
                        axs[2].axis('off')

                        plt.tight_layout()
                        # Save the plot to the ./plots directory
                        plt.savefig(f'./plots/epoch_{epoch+1}_image_{i+1}.png')
                        plt.close()

                    # Save the model every 10 epochs
                    model_dir = 'models'
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, f'model_epoch_{epoch+1}.pth')
                    # Save model.module.state_dict() to account for DDP wrapping
                    torch.save(model.module.state_dict(), model_path)
                    print(f"Saved model at {model_path}")

        # Synchronize all processes
        dist.barrier()
        dist.destroy_process_group()
        print(f"Rank {rank} finished training.")

    except Exception as e:
        print(f"Rank {rank} encountered an exception: {e}")
        dist.destroy_process_group()

# ============================
# Entry Point
# ============================

if __name__ == '__main__':
    world_size = torch.cuda.device_count()  # Use all available GPUs
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
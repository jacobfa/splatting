#!/usr/bin/env python3

import os
import math
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

# For progress bar
from tqdm import tqdm

# For plotting
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (useful on servers)
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 1) Random hole mask helper
# -----------------------------------------------------------------------------
def random_mask(img, hole_size=64):
    """
    Given a tensor image of shape (3, H, W),
    create a random square hole and return (partial_img, mask).
    mask: (1, H, W) with 1s for known pixels, 0s for the hole.
    partial_img = original * mask
    """
    _, h, w = img.shape
    if hole_size >= h or hole_size >= w:
        raise ValueError(f"Hole size ({hole_size}) is too large for image {h}x{w}.")

    top = torch.randint(0, h - hole_size, (1,)).item()
    left = torch.randint(0, w - hole_size, (1,)).item()
    
    mask = torch.ones((1, h, w), dtype=img.dtype)
    mask[:, top:top+hole_size, left:left+hole_size] = 0
    
    partial_img = img.clone()
    partial_img[:, top:top+hole_size, left:left+hole_size] = 0.0
    
    return partial_img, mask


# -----------------------------------------------------------------------------
# 2) Example inpainting dataset (using torchvision's Places365 for demonstration)
# -----------------------------------------------------------------------------
class Places365InpaintingDataset(Dataset):
    """
    Wraps torchvision.datasets.Places365 to apply random hole masks.
    By default, uses 'train-standard' if train=True, else 'val'.
    """
    def __init__(self, root='.', train=True, transform=None, hole_size=64, small=False):
        super().__init__()
        self.hole_size = hole_size
        self.train = train
        split = 'train-standard' if train else 'val'

        self.dataset = datasets.Places365(
            root=root,
            split=split,
            small=small,       # If True, images are ~256 on shorter side
            download=True,     # Will attempt to download metadata/devkit
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # shape: (3, H, W) after transform
        partial_img, mask = random_mask(img, self.hole_size)
        gt_img = img  # ground truth is the original
        return partial_img, mask, gt_img


# -----------------------------------------------------------------------------
# 3) A UNet backbone for diffusion (3 downs, 1 bottleneck, 3 ups => final 256x256)
# -----------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """
    (Conv => ReLU => Conv => ReLU)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscale by stride-2 conv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),  # stride=2
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upscale by transpose conv, then concatenate skip connection, then DoubleConv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Transpose conv
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.act = nn.ReLU(inplace=True)
        # After concatenation with the skip, channel count is out_ch + skip_ch => 2*out_ch if skip is out_ch
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.act(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class SimpleUNet(nn.Module):
    """
    UNet with 3 downs / 3 ups. For a 256x256 input, the shape changes as:
       down0: 256 -> 128
       down1: 128 -> 64
       down2: 64 -> 32
    Bottleneck => shape 32x32
       up2: 32 -> 64
       up1: 64 -> 128
       up0: 128 -> 256
    Output => 256x256
    """
    def __init__(self, in_channels=7, base_channels=64):
        """
        in_channels=7:
         - 3 for the current noisy image
         - 4 for partial_img (3) + mask (1)
        """
        super().__init__()
        # Encoder
        self.down0 = Down(in_channels, base_channels)         # => (64, 128x128)
        self.down1 = Down(base_channels, base_channels * 2)   # => (128, 64x64)
        self.down2 = Down(base_channels * 2, base_channels * 4) # => (256, 32x32)

        # Bottleneck
        self.mid = DoubleConv(base_channels * 4, base_channels * 4)

        # Decoder
        self.up2 = Up(base_channels * 4, base_channels * 2)  # skip from down1
        self.up1 = Up(base_channels * 2, base_channels)      # skip from down0

        # We only have 3 Down blocks, so we do 2 Up blocks so far. We need a final Up:
        self.up0 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.final_conv = DoubleConv(base_channels, base_channels)

        # Output
        self.out = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, x_noisy, x_cond):
        """
        x_noisy: (B, 3, H, W)
        x_cond:  (B, 4, H, W)
         => cat => (B,7,H,W)
        We'll produce (B,3,H,W) as the predicted noise.
        """
        inp = torch.cat([x_noisy, x_cond], dim=1)  # => (B,7,H,W)

        # Encoder
        d0 = self.down0(inp)   # => (64, H/2, W/2)
        d1 = self.down1(d0)    # => (128, H/4, W/4)
        d2 = self.down2(d1)    # => (256, H/8, W/8)

        # Bottleneck
        m = self.mid(d2)       # => (256, H/8, W/8)

        # Decoder
        # 1) up2 from (256->128), skip from d1(128)
        u2 = self.up2(m, d1)   # => (128, H/4, W/4)
        # 2) up1 from (128->64), skip from d0(64)
        u1 = self.up1(u2, d0)  # => (64, H/2, W/2)

        # 3) final up0 from 64->64 channels + spatial x2 => (64, H, W)
        u0 = self.up0(u1)      # => (64, H, W)
        u0 = self.final_conv(u0)  # => (64, H, W)

        out = self.out(u0)     # => (3, H, W)
        return out


# -----------------------------------------------------------------------------
# 4) DDPM-like diffusion utilities
# -----------------------------------------------------------------------------
def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    """
    Linear schedule from start to end over 'timesteps'.
    """
    return torch.linspace(start, end, timesteps)


class DiffusionModel:
    """
    Simple DDPM-like diffusion for inpainting.
    We'll do standard training on entire images but *condition* on partial image + mask.
    Then for sampling (inference), clamp the known region at each step.
    """
    def __init__(self, net: nn.Module, timesteps=1000, device='cuda'):
        self.net = net
        self.timesteps = timesteps
        self.device = device

        # Create beta schedule
        self.betas = linear_beta_schedule(timesteps, 1e-4, 0.02).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For sampling
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x0, t, noise=None):
        """
        Diffusion forward process:
          q(x_t | x_0) = N(x_t; sqrt_alphas_cumprod[t]*x_0, (1 - alphas_cumprod[t])I)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_1m_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return sqrt_alpha_t * x0 + sqrt_1m_alpha_t * noise

    def p_losses(self, x0, t, cond, mask_weight=1.0):
        """
        Training loss: we predict the noise that was added.
          x0: original clean image  (B,3,H,W)
          t: diffusion timestep
          cond: condition (partial_img + mask), shape (B,4,H,W)
          mask_weight: optional weighting for the masked region

        We create random noise, generate x_t, then the net predicts that noise.
        Loss = MSE(predicted_noise, actual_noise).
        """
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)  # shape => (B,3,H,W)

        # Predict noise with UNet
        pred_noise = self.net(x_t, cond)         # also (B,3,H,W)

        # MSE over all pixels
        mse_per_pixel = F.mse_loss(pred_noise, noise, reduction='none')  # (B,3,H,W)

        # Optionally emphasize the masked region
        if mask_weight != 1.0 and cond.size(1) == 4:
            real_mask = cond[:, 3:4, :, :]   # shape (B,1,H,W), 1=known, 0=hole
            hole_mask = 1.0 - real_mask
            hole_mask = hole_mask.repeat(1,3,1,1)
            weighted_mse = mse_per_pixel * (1.0 + hole_mask * (mask_weight - 1.0))
            return weighted_mse.mean()
        else:
            return mse_per_pixel.mean()

    @torch.no_grad()
    def p_sample(self, x_t, t, cond):
        """
        One step of reverse diffusion:
          p(x_{t-1} | x_t) ~ N(mean, var)
          mean = (1/sqrt(alpha_t)) [ x_t - ( (1-alpha_t)/sqrt(1-alpha_bar_t) ) * model_noise ]
          var = posterior_variance[t]

        Then clamp known region from cond (partial_img) if mask=1.
        """
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = 1.0 / torch.sqrt(self.alphas[t])
        posterior_var_t = self.posterior_variance[t]

        # model predicts noise
        model_noise = self.net(x_t, cond)

        # formula
        x_pred = sqrt_recip_alphas_t * (
            x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * model_noise
        )

        if t > 0:
            z = torch.randn_like(x_t)
        else:
            z = 0.0  # last step, no noise

        x_t_next = x_pred + torch.sqrt(posterior_var_t) * z

        # Hard inpainting step: clamp the known region
        partial_img = cond[:, :3, :, :]
        known_mask = cond[:, 3:4, :, :]
        x_t_next = known_mask * partial_img + (1.0 - known_mask) * x_t_next

        return x_t_next

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        """
        Reverse diffusion from x_T ~ N(0,1) down to x_0.
        shape: (B, 3, H, W)
        cond:  (B, 4, H, W)
        """
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            x = self.p_sample(x, i, cond)
        return x


# -----------------------------------------------------------------------------
# 5) Training one epoch
# -----------------------------------------------------------------------------
def train_one_epoch(diffusion, model, dataloader, optimizer, device, max_grad_norm=1.0):
    """
    diffusion: the DiffusionModel object
    model: the underlying UNet (model.net)
    """
    model.train()
    total_loss = 0.0
    count = 0

    for partial_img, mask, gt_img in tqdm(dataloader, desc="Training", leave=False):
        partial_img = partial_img.to(device)  # (B,3,H,W)
        mask = mask.to(device)               # (B,1,H,W)
        gt_img = gt_img.to(device)           # (B,3,H,W)

        # Condition: cat(partial_img, mask) => (B,4,H,W)
        cond = torch.cat([partial_img, mask], dim=1)

        # Random t
        t = torch.randint(0, diffusion.timesteps, (partial_img.size(0),), device=device).long()

        optimizer.zero_grad()

        # Diffusion loss
        loss = diffusion.p_losses(gt_img, t, cond, mask_weight=2.0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / count


# -----------------------------------------------------------------------------
# 6) Function to save inpainting examples (sampling)
# -----------------------------------------------------------------------------
@torch.no_grad()
def sample_inpainting_examples(diffusion, dataset, device, epoch, num_examples=3):
    """
    We'll pick random examples from the dataset, do reverse diffusion,
    and save a grid: (Original, Partial, Inpainted).
    """
    indices = torch.randint(len(dataset), size=(num_examples,))
    fig, axes = plt.subplots(num_examples, 3, figsize=(10, 3 * num_examples))

    diffusion.net.eval()
    
    for i, idx in enumerate(indices):
        partial_img, mask, gt_img = dataset[idx]
        partial_img = partial_img.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        gt_img = gt_img.unsqueeze(0).to(device)

        _, h, w = gt_img.shape[-3:]  # (3,H,W)

        # Condition
        cond = torch.cat([partial_img, mask], dim=1)  # (1,4,H,W)

        # Reverse diffusion
        x_gen = diffusion.p_sample_loop(shape=(1,3,h,w), cond=cond)
        final_img = x_gen  # known region is clamped inside p_sample_loop

        # Move to CPU for plotting
        partial_img = partial_img.cpu().squeeze()
        gt_img = gt_img.cpu().squeeze()
        final_img = final_img.cpu().squeeze()

        # Plot original
        axes[i, 0].imshow(gt_img.permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        # Plot partial
        axes[i, 1].imshow(partial_img.permute(1, 2, 0).numpy())
        axes[i, 1].set_title("Partial")
        axes[i, 1].axis('off')

        # Plot final
        axes[i, 2].imshow(final_img.permute(1, 2, 0).numpy())
        axes[i, 2].set_title("Inpainted")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    out_name = f"diffusion_inpaint_epoch_{epoch}.png"
    plt.savefig(out_name)
    plt.close()
    print(f"Saved diffusion inpainting results to {out_name}")


# -----------------------------------------------------------------------------
# 7) Main function
# -----------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="Places365", 
                        help="Root directory for Places365.")
    parser.add_argument("--train", action="store_true", 
                        help="Use the 'train-standard' split. Otherwise uses 'val'.")
    parser.add_argument("--small", action="store_true",
                        help="Use the 'small' version of Places365 (256x256).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--img_size", type=int, default=256, 
                        help="Resize images to this size if using the large dataset.")
    parser.add_argument("--hole_size", type=int, default=64, help="Square hole size for inpainting.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # If using the big dataset, let's definitely transform to 256x256. If 'small' is True, 
    # they're already ~256 but we can still enforce transforms.
    transform_list = []
    if not args.small:
        transform_list.append(transforms.Resize((args.img_size, args.img_size)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    # Dataset & DataLoader
    dataset = Places365InpaintingDataset(
        root=args.root,
        train=args.train,
        transform=transform,
        hole_size=args.hole_size,
        small=args.small
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    # Model + Diffusion
    unet = SimpleUNet(in_channels=7, base_channels=64).to(device)
    diffusion = DiffusionModel(net=unet, timesteps=args.timesteps, device=device)

    # Optimizer
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    loss_history = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch [{epoch}/{args.epochs}] ---")
        train_loss = train_one_epoch(
            diffusion, unet, dataloader, optimizer, device,
            max_grad_norm=args.max_grad_norm
        )
        loss_history.append(train_loss)

        logging.info(f"Epoch [{epoch}/{args.epochs}]  Diffusion Loss: {train_loss:.4f}")

        # Save inpainting samples
        sample_inpainting_examples(diffusion, dataset, device, epoch, num_examples=3)

        # Save checkpoint
        checkpoint_path = f"places365_diffusion_checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history
        }, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")

    # Plot final training loss
    plt.figure()
    plt.plot(loss_history, label='Diffusion Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Diffusion Inpainting on Places365)')
    plt.legend()
    plt.savefig('places365_diffusion_loss_plot.png')
    plt.close()
    logging.info("Training complete. Loss plot saved to places365_diffusion_loss_plot.png.")


if __name__ == "__main__":
    main()

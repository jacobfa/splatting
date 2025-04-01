#!/usr/bin/env python3

import os
import math
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils

# For progress bar
from tqdm import tqdm

# For plotting
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (useful on servers)
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# 1) Random hole mask helper
# -------------------------------------------------------------------------
def random_mask(img, hole_size=8):
    """
    Given a tensor image of shape (3, H, W),
    create a random square hole and return (partial_img, mask).
    mask: (1, H, W) with 1s for known, 0s for the hole.
    partial_img = original * mask
    """
    _, h, w = img.shape
    top = torch.randint(0, h - hole_size, (1,)).item()
    left = torch.randint(0, w - hole_size, (1,)).item()
    
    mask = torch.ones((1, h, w), dtype=img.dtype)
    mask[:, top:top+hole_size, left:left+hole_size] = 0
    
    partial_img = img.clone()
    partial_img[:, top:top+hole_size, left:left+hole_size] = 0.0
    
    return partial_img, mask


# -------------------------------------------------------------------------
# 2) CIFAR-10 inpainting dataset
# -------------------------------------------------------------------------
class CIFAR10InpaintingDataset(Dataset):
    def __init__(self, root='data', train=True, transform=None, hole_size=8):
        super().__init__()
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
        self.hole_size = hole_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # shape: (3, 32, 32)
        partial_img, mask = random_mask(img, self.hole_size)
        # ground truth is the original
        gt_img = img
        return partial_img, mask, gt_img


# -------------------------------------------------------------------------
# 3) A simple Transformer-based generator
#    We'll do patchify => transformer => unpatchify
# -------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """
    Splits an image into patches, then projects each patch to an embed dimension.
    For a 32x32 image with patch_size=4 => 8x8=64 patches total.
    """
    def __init__(self, in_chans=4, embed_dim=192, patch_size=4, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # Project each patch (in_chans * patch_size * patch_size) -> embed_dim
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Number of patches
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x):
        # x: (B, in_chans, 32, 32)
        x = self.proj(x)  # => (B, embed_dim, 8, 8) for patch_size=4
        B, C, H, W = x.shape
        x = x.flatten(2)   # (B, C, H*W) => (B, 192, 64)
        x = x.transpose(1, 2)  # => (B, 64, 192)
        return x


class PatchUnembed(nn.Module):
    """
    Reverse of PatchEmbed: take (B, num_patches, embed_dim) and reshape
    back to (B, out_chans, img_size, img_size).
    """
    def __init__(self, out_chans=3, embed_dim=192, patch_size=4, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        self.conv = nn.ConvTranspose2d(
            embed_dim,
            out_chans,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x, B):
        # x: (B, num_patches, embed_dim)
        num_patches_per_row = self.img_size // self.patch_size

        x = x.transpose(1, 2)  # => (B, embed_dim, num_patches)
        x = x.view(B, self.embed_dim, num_patches_per_row, num_patches_per_row)
        x = self.conv(x)  # => (B, out_chans, 32, 32)
        return x


class SimpleTransformerGenerator(nn.Module):
    """
    Input: (partial_img + mask) => patchify => Transformer => unpatchify => (full image).
    """
    def __init__(self, in_chans=4, out_chans=3, 
                 img_size=32, patch_size=4, 
                 embed_dim=192, depth=4, num_heads=6):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size, img_size)
        
        # Positional embeddings for each patch
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=depth
        )

        # Unpatchify
        self.unpatch = PatchUnembed(out_chans, embed_dim, patch_size, img_size)

    def forward(self, x):
        # x shape: (B, 4, 32, 32)
        B = x.shape[0]
        x = self.patch_embed(x)          # => (B, num_patches, embed_dim)
        x = x + self.pos_embed           # Add positional embedding
        x = self.transformer(x)          # => (B, num_patches, embed_dim)
        out = self.unpatch(x, B)         # => (B, out_chans, 32, 32)
        return out


# -------------------------------------------------------------------------
# 4) Discriminator (simple CNN)
# -------------------------------------------------------------------------
class SimpleDiscriminator(nn.Module):
    """
    Patch-based CNN for real/fake classification.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels*2, 1, 4, 1, 1)
        )

    def forward(self, x):
        # x: (B, 3, 32, 32)
        return self.net(x)


# -------------------------------------------------------------------------
# 5) Utility losses
# -------------------------------------------------------------------------
def adversarial_loss(pred, is_real=True):
    """
    Minimizes BCE loss for real/fake classification.
    """
    target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return F.binary_cross_entropy_with_logits(pred, target)

def reconstruction_loss(pred, target, mask):
    """
    L1 loss in the known (mask=1) region only.
    """
    return F.l1_loss(pred * mask, target * mask)


# -------------------------------------------------------------------------
# 6) Training function (uses tqdm + gradient clipping)
# -------------------------------------------------------------------------
def train_one_epoch(gen, disc, dataloader, optG, optD, device,
                    lambda_rec=1.0, lambda_adv=0.01, max_grad_norm=1.0):
    """
    Runs one epoch of training with TQDM progress bar.
    Returns average G and D loss.
    """
    gen.train()
    disc.train()

    total_g_loss = 0.0
    total_d_loss = 0.0
    count = 0

    # Wrap the dataloader with tqdm for a batch progress bar
    for partial_img, mask, gt_img in tqdm(dataloader, desc="Training", leave=False):
        partial_img = partial_img.to(device)
        mask = mask.to(device)
        gt_img = gt_img.to(device)

        # -------------------
        # Train Discriminator
        # -------------------
        optD.zero_grad()
        with torch.no_grad():
            pred_img = gen(torch.cat([partial_img, mask], dim=1))

        d_real = disc(gt_img)
        loss_d_real = adversarial_loss(d_real, is_real=True)

        d_fake = disc(pred_img.detach())
        loss_d_fake = adversarial_loss(d_fake, is_real=False)

        lossD = 0.5 * (loss_d_real + loss_d_fake)

        # Check for NaN
        if torch.isnan(lossD):
            print("Warning: D loss is NaN. Skipping D update.")
        else:
            lossD.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(disc.parameters(), max_grad_norm)
            optD.step()

        # -------------------
        # Train Generator
        # -------------------
        optG.zero_grad()
        pred_img = gen(torch.cat([partial_img, mask], dim=1))
        loss_rec = reconstruction_loss(pred_img, gt_img, mask)

        d_fake_for_g = disc(pred_img)
        loss_adv = adversarial_loss(d_fake_for_g, is_real=True)

        lossG = lambda_rec * loss_rec + lambda_adv * loss_adv

        if torch.isnan(lossG):
            print("Warning: G loss is NaN. Skipping G update.")
        else:
            lossG.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(gen.parameters(), max_grad_norm)
            optG.step()

        total_d_loss += lossD.item()
        total_g_loss += lossG.item()
        count += 1

    avg_d_loss = total_d_loss / count
    avg_g_loss = total_g_loss / count

    return avg_g_loss, avg_d_loss


# -------------------------------------------------------------------------
# 7) Function to save example original/partial/reconstructed images
# -------------------------------------------------------------------------
def save_inpainting_examples(gen, dataset, device, epoch, num_examples=5, hole_size=8):
    """
    Saves a grid of images showing partial images and their reconstructions.
    num_examples = number of examples to visualize.
    """
    gen.eval()
    
    # We'll just pick a few random indices
    indices = torch.randint(len(dataset), size=(num_examples,))
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(6, 2 * num_examples))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            partial_img, mask, gt_img = dataset[idx]
            # Move to device
            partial_img = partial_img.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            gt_img = gt_img.unsqueeze(0).to(device)

            # Generate inpainted image
            pred_img = gen(torch.cat([partial_img, mask], dim=1))  # (B, 3, 32, 32)

            # Move to CPU for plotting
            partial_img = partial_img.cpu()
            gt_img = gt_img.cpu()
            pred_img = pred_img.cpu()
            
            # Plot original
            axes[i, 0].imshow(gt_img.squeeze().permute(1, 2, 0).numpy())
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')
            
            # Plot partial
            axes[i, 1].imshow(partial_img.squeeze().permute(1, 2, 0).numpy())
            axes[i, 1].set_title("Partial")
            axes[i, 1].axis('off')

            # Plot prediction
            axes[i, 2].imshow(pred_img.squeeze().permute(1, 2, 0).numpy())
            axes[i, 2].set_title("Inpainted")
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    out_name = f"examples_epoch_{epoch}.png"
    plt.savefig(out_name)
    plt.close()
    print(f"Saved example inpainting results to {out_name}")


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Models
    gen = SimpleTransformerGenerator(
        in_chans=4, out_chans=3,
        img_size=32, patch_size=4,
        embed_dim=192, depth=4, num_heads=6
    ).to(device)

    disc = SimpleDiscriminator(
        in_channels=3, base_channels=64
    ).to(device)

    # Optimizers
    optG = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Optional scheduler for improved training stability
    schedulerG = torch.optim.lr_scheduler.StepLR(optG, step_size=3, gamma=0.5)
    schedulerD = torch.optim.lr_scheduler.StepLR(optD, step_size=3, gamma=0.5)

    # Dataset & DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR10InpaintingDataset(
        root="./data",
        train=True,
        transform=transform,
        hole_size=8
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    g_loss_history = []
    d_loss_history = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch [{epoch}/{args.epochs}] ---")
        g_loss, d_loss = train_one_epoch(
            gen, disc, dataloader, optG, optD, device,
            lambda_rec=1.0, lambda_adv=0.01,
            max_grad_norm=args.max_grad_norm
        )
        g_loss_history.append(g_loss)
        d_loss_history.append(d_loss)

        logging.info(f"Epoch [{epoch}/{args.epochs}]  G Loss: {g_loss:.4f}  D Loss: {d_loss:.4f}")

        # Update learning rate
        schedulerG.step()
        schedulerD.step()

        # Save a few example reconstructions from this epoch
        save_inpainting_examples(gen, dataset, device, epoch, num_examples=5, hole_size=8)

        # Optionally save a checkpoint each epoch
        checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'optG_state_dict': optG.state_dict(),
            'optD_state_dict': optD.state_dict(),
            'g_loss_history': g_loss_history,
            'd_loss_history': d_loss_history
        }, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")

    # Plot final G and D losses
    plt.figure()
    plt.plot(g_loss_history, label='Generator Loss')
    plt.plot(d_loss_history, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()
    logging.info("Training complete. Loss plot saved to 'loss_plot.png'.")


if __name__ == "__main__":
    main()

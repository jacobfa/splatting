#!/usr/bin/env python3

import os
import math
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, distributed
from torchvision import datasets, transforms

# -------------------------------------------------------------------------
# 1) Random hole mask helper
# -------------------------------------------------------------------------
def random_mask(img, hole_size=8):
    """
    Given a tensor image of shape (3, H, W),
    create a random square hole and return (partial_img, mask).
    mask: (1, H, W) with 1s for known, 0s for the hole
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
        img, _ = self.dataset[idx]  # img shape: (3, 32, 32)
        partial_img, mask = random_mask(img, self.hole_size)
        # ground truth is the original
        gt_img = img
        return partial_img, mask, gt_img


# -------------------------------------------------------------------------
# 3) A simple Transformer-based generator
#    We'll do a patchify => transformer => unpatchify approach
# -------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """
    Splits an image into patches, and projects each patch to an embed dimension.
    For a 32x32 image, you could do e.g. 4x4 patches => 8x8=64 patches total.
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
        # out: (B, embed_dim, #patches_h, #patches_w)
        x = self.proj(x)  # => (B, embed_dim, 32/patch_size, 32/patch_size)
        # Flatten the spatial dimensions
        B, C, H, W = x.shape
        x = x.flatten(2)   # (B, C, H*W)
        x = x.transpose(1, 2)  # (B, H*W, C)
        return x  # shape (B, num_patches, embed_dim)

class PatchUnembed(nn.Module):
    """
    Inverse of PatchEmbed: take (B, num_patches, embed_dim) and reshape
    back to an image of shape (B, out_chans, img_size, img_size).
    """
    def __init__(self, out_chans=3, embed_dim=192, patch_size=4, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        
        self.conv = nn.ConvTranspose2d(
            embed_dim,
            out_chans,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x, B):
        # x: (B, num_patches, embed_dim)
        # Reshape to (B, embed_dim, #patches_h, #patches_w) => then convTranspose
        n_patches = x.shape[1]
        num_patches_per_row = self.img_size // self.patch_size

        x = x.transpose(1, 2)  # => (B, embed_dim, num_patches)
        x = x.view(B, self.embed_dim, num_patches_per_row, num_patches_per_row)
        x = self.conv(x)  # => (B, out_chans, 32, 32)
        return x

class SimpleTransformerGenerator(nn.Module):
    """
    Takes as input (partial_img + mask) => uses patchify => Transformer => unpatchify => (full image).
    We'll use a single standard TransformerEncoder for demonstration.
    """
    def __init__(self, in_chans=4, out_chans=3, img_size=32, patch_size=4, embed_dim=192, depth=4, num_heads=6):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size, img_size)
        
        # Positional embeddings for each patch
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Unpatchify (map embed_dim => out_chans)
        self.unpatch = PatchUnembed(out_chans, embed_dim, patch_size, img_size)

    def forward(self, x):
        # x shape: (B, 4, 32, 32)
        B = x.shape[0]
        # Patchify
        x = self.patch_embed(x)  # => (B, num_patches, embed_dim)
        # Add positional embedding
        x = x + self.pos_embed
        # Transformer
        x = self.transformer(x)  # => (B, num_patches, embed_dim)
        # Unpatchify
        out = self.unpatch(x, B)  # => (B, out_chans, 32, 32)
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
        return self.net(x)


# -------------------------------------------------------------------------
# 5) Utility losses and training
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


def train_one_epoch(rank, world_size, epoch, gen, disc,
                    dataloader, optG, optD,
                    lambda_rec=1.0, lambda_adv=0.01):
    """
    Runs one epoch of training. Logs stats once per epoch (rank=0).
    """
    # Switch to train mode
    gen.train()
    disc.train()

    # We'll collect the total or average losses
    total_g_loss = 0.0
    total_d_loss = 0.0
    count = 0

    for (partial_img, mask, gt_img) in dataloader:
        partial_img = partial_img.cuda(rank, non_blocking=True)
        mask = mask.cuda(rank, non_blocking=True)
        gt_img = gt_img.cuda(rank, non_blocking=True)

        # -------------------
        # Train Discriminator
        # -------------------
        optD.zero_grad()
        with torch.no_grad():
            # Generator forward
            pred_img = gen(torch.cat([partial_img, mask], dim=1))
        # Real
        d_real = disc(gt_img)
        loss_d_real = adversarial_loss(d_real, is_real=True)
        # Fake
        d_fake = disc(pred_img.detach())
        loss_d_fake = adversarial_loss(d_fake, is_real=False)
        lossD = 0.5*(loss_d_real + loss_d_fake)
        lossD.backward()
        optD.step()

        # -------------------
        # Train Generator
        # -------------------
        optG.zero_grad()
        pred_img = gen(torch.cat([partial_img, mask], dim=1))
        # reconstruction
        loss_rec = reconstruction_loss(pred_img, gt_img, mask)
        # adversarial
        d_fake_for_g = disc(pred_img)
        loss_adv = adversarial_loss(d_fake_for_g, is_real=True)
        # total
        lossG = lambda_rec*loss_rec + lambda_adv*loss_adv
        lossG.backward()
        optG.step()

        # Accumulate
        total_d_loss += lossD.item()
        total_g_loss += lossG.item()
        count += 1

    # Average over all batches
    avg_d_loss = total_d_loss / count
    avg_g_loss = total_g_loss / count

    # If rank=0, log to log.log
    if rank == 0:
        logging.info(f"Epoch {epoch} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")
    return avg_g_loss, avg_d_loss


# -------------------------------------------------------------------------
# 6) Main entry point for DDP
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # Initialize process group (NCCL)
    dist.init_process_group(backend="nccl", init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Force each process to only see 1 GPU (the one assigned by torchrun)
    torch.cuda.set_device(args.local_rank)

    # Only rank=0 logs to file
    if rank == 0:
        # Overwrite previous log.log each run
        logging.basicConfig(filename="log.log", filemode='w', level=logging.INFO)
        logging.info(f"World Size: {world_size}")
    else:
        # Ranks != 0 don't log
        logging.basicConfig(filename=os.devnull, level=logging.CRITICAL)

    # -----------------------
    # Create model + optimizer
    # -----------------------
    gen = SimpleTransformerGenerator(
        in_chans=4, out_chans=3,
        img_size=32, patch_size=4,
        embed_dim=192, depth=4, num_heads=6
    ).cuda(args.local_rank)

    disc = SimpleDiscriminator(
        in_channels=3, base_channels=64
    ).cuda(args.local_rank)

    # Wrap in DDP
    ddp_gen = DDP(gen, device_ids=[args.local_rank], output_device=args.local_rank)
    ddp_disc = DDP(disc, device_ids=[args.local_rank], output_device=args.local_rank)

    optG = torch.optim.Adam(ddp_gen.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(ddp_disc.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # -----------------------
    # Create Dataset & DLoader
    # -----------------------
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CIFAR10InpaintingDataset(
        root="./data",
        train=True,
        transform=transform,
        hole_size=8
    )

    # Distributed Sampler
    sampler = distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=0,       # No multiprocessing
        pin_memory=True,
        drop_last=True
    )

    # -----------------------
    # Training loop
    # -----------------------
    epochs = 5
    for epoch in range(1, epochs+1):
        sampler.set_epoch(epoch)  # ensure each epoch sees different samples across replicas
        train_one_epoch(rank, world_size, epoch,
                        ddp_gen, ddp_disc,
                        dataloader, optG, optD,
                        lambda_rec=1.0, lambda_adv=0.01)

    # Cleanup
    dist.destroy_process_group()
    if rank == 0:
        logging.info("Training complete. Exiting.")


if __name__ == "__main__":
    main()

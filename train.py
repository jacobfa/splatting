import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

def create_meshgrid(height, width, device='cpu'):
    """
    Returns a meshgrid of pixel coordinates: shape (2, H, W),
    where grid[0] is x-coord, grid[1] is y-coord.
    """
    ys, xs = torch.meshgrid(
        torch.arange(0, height, device=device),
        torch.arange(0, width, device=device),
        indexing='ij'
    )
    grid = torch.stack([xs, ys], dim=0)  # shape: (2, H, W)
    return grid

def safe_exp(x, clamp_val=10.0):
    """
    A 'safe' exponential that clamps input to avoid overflow.
    """
    return torch.exp(torch.clamp(x, max=clamp_val))

class AnisotropicInpaintNet(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64):
        """
        Args:
            in_channels: e.g., 3 for RGB + 1 for mask = 4
            hidden_channels: # of hidden features in CNN
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        
        # Final layer outputs 6 channels per pixel:
        #   (ell_11, ell_21, ell_22, c_r, c_g, c_b)
        self.conv_out = nn.Conv2d(hidden_channels, 6, kernel_size=1, padding=0)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, 4, H, W) = [RGB_image + mask]
        Returns:
            (B, 6, H, W) = [ell_11, ell_21, ell_22, c_r, c_g, c_b]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        param_out = self.conv_out(x)
        return param_out

def render_anisotropic_gaussians(param_out, mask, device='cpu',
                                 truncate_thresh=1e-4, chunk_size=128):
    """
    Renders the final inpainted image by splatting anisotropic Gaussians,
    using a chunked vectorized approach for faster performance.

    Args:
        param_out: (B, 6, H, W) from the network
                   [ell_11, ell_21, ell_22, c_r, c_g, c_b]
        mask:      (B, 1, H, W) binary mask (1=known, 0=hole)
        device:    torch device
        truncate_thresh: min weight threshold
        chunk_size:      how many "center pixels" to process at once
                         (tune to manage memory/speed)
    Returns:
        (B, 3, H, W) inpainted image
    """
    B, _, H, W = param_out.shape
    
    # 1) Separate and flatten predicted parameters
    ell_11 = param_out[:, 0, :, :].view(B, -1)   # (B, H*W)
    ell_21 = param_out[:, 1, :, :].view(B, -1)
    ell_22 = param_out[:, 2, :, :].view(B, -1)
    c_rgb  = param_out[:, 3:6, :, :].view(B, 3, -1)  # (B, 3, H*W)
    
    # 2) Flatten coordinate grid
    grid = create_meshgrid(H, W, device=device)  # shape (2, H, W)
    xs_flat = grid[0].view(-1)  # (H*W,)
    ys_flat = grid[1].view(-1)

    # 3) Prepare accumulators for color and weight
    weighted_color_acc = torch.zeros((B, 3, H, W), device=device)
    weight_acc = torch.zeros((B, 1, H, W), device=device)

    n_pixels = H * W
    # 4) Process pixel centers in chunks to reduce memory usage
    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)
        size = end - start
        
        # 4a) Gather the relevant param slices for these centers
        L_11 = ell_11[:, start:end]   # (B, size)
        L_21 = ell_21[:, start:end]
        L_22 = ell_22[:, start:end]
        c_chunk = c_rgb[:, :, start:end]  # (B, 3, size)
        
        # 4b) Compute the inverse covariance for each center
        #     Sigma = L L^T => we invert L, then build M = inv(Sigma).
        denom = L_11 * L_22  # shape (B, size)
        eps = 1e-5
        denom = denom.clamp(min=eps)
        L_inv_11 = (L_22 / denom).view(B, size, 1, 1)
        L_inv_21 = (-L_21 / denom).view(B, size, 1, 1)
        L_inv_22 = (L_11 / denom).view(B, size, 1, 1)

        # M = L_inv^T @ L_inv, done directly:
        # shape => (B, size, 1, 1)
        M_00 = L_inv_11**2 + L_inv_21**2
        M_01 = L_inv_21 * L_inv_22
        M_10 = L_inv_22 * L_inv_21
        M_11 = L_inv_22**2

        # 4c) For each center, compute dx, dy over the entire image
        #     dx, dy shapes => (1, size, H, W), then broadcast to (B, size, H, W)
        dx = grid[0].view(1, 1, H, W) - xs_flat[start:end].view(1, size, 1, 1)
        dy = grid[1].view(1, 1, H, W) - ys_flat[start:end].view(1, size, 1, 1)
        dx = dx.to(device)
        dy = dy.to(device)

        # Expand for the batch dimension (B)
        dx = dx.expand(B, -1, -1, -1)  # => (B, size, H, W)
        dy = dy.expand(B, -1, -1, -1)  # => (B, size, H, W)

        # 4d) Evaluate the exponent: quad = [dx, dy] * M * [dx, dy]^T
        #     M_00, M_10, etc. are (B, size, 1, 1)
        Wx = M_00 * dx + M_10 * dy    # => (B, size, H, W)
        Wy = M_01 * dx + M_11 * dy
        quad = Wx * dx + Wy * dy      # => (B, size, H, W)

        w_i = torch.exp(-0.5 * quad)
        w_i = torch.where(w_i < truncate_thresh, torch.zeros_like(w_i), w_i)

        # 4e) Multiply by color
        # c_chunk => (B, 3, size)
        # we want shape => (B, size, 3, 1, 1) to broadcast
        c_broadcast = c_chunk.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        # => (B, size, 3, 1, 1)

        # Weighted color => (B, size, 3, H, W)
        weighted_color = w_i.unsqueeze(2) * c_broadcast

        # Sum over the "size" dimension
        weighted_color_sum = weighted_color.sum(dim=1)  # => (B, 3, H, W)
        weight_sum = w_i.sum(dim=1, keepdim=True)       # => (B, 1, H, W)

        # Accumulate into global buffers
        weighted_color_acc += weighted_color_sum
        weight_acc += weight_sum

    # 5) Final image = weighted sum / total weight
    inpainted = torch.where(
        weight_acc > 1e-8,
        weighted_color_acc / (weight_acc + 1e-8),
        torch.zeros_like(weighted_color_acc)
    )

    return inpainted

def reconstruction_loss(pred, target, mask):
    """
    L1 on the known region only (where mask=1).
    """
    return F.l1_loss(pred * mask, target * mask)

def smoothness_loss(ell_11, ell_21, ell_22):
    """
    Encourages smoothness in the Cholesky parameters.
    A simple approach: L1 of local gradients.
    """
    def gradient_horiz(x):
        return x[..., 1:] - x[..., :-1]
    
    def gradient_vert(x):
        return x[..., 1:, :] - x[..., :-1, :]
    
    loss_val = 0
    for ell in [ell_11, ell_21, ell_22]:
        dx = gradient_horiz(ell)
        dy = gradient_vert(ell)
        loss_val += (dx.abs().mean() + dy.abs().mean())
    return loss_val

class SimpleDiscriminator(nn.Module):
    """
    A placeholder patch‐based discriminator (GAN).
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

def adversarial_loss(pred, is_real=True):
    """
    Minimizes BCE loss for real/fake classification.
    """
    target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return F.binary_cross_entropy_with_logits(pred, target)

def train_epoch(netG, netD, train_loader, optimizerG, optimizerD, device='cpu',
                lambda_rec=1.0, lambda_smooth=0.1, lambda_adv=0.01):
    """
    Runs one training epoch.
    Uses tqdm to display a progress bar and returns batch losses (for plotting).
    """
    netG.train()
    netD.train()

    # Lists to track losses per batch
    batch_losses_rec = []
    batch_losses_smooth = []
    batch_losses_adv = []
    batch_losses_d = []
    
    loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
    
    for partial_img, mask, gt_img in loader_tqdm:
        partial_img = partial_img.to(device)  # (B, 3, H, W)
        mask = mask.to(device)                # (B, 1, H, W)
        gt_img = gt_img.to(device)            # (B, 3, H, W)
        
        # Concatenate the mask as an extra channel => (B, 4, H, W)
        input_netG = torch.cat([partial_img, mask], dim=1)
        
        # Generator forward
        param_out = netG(input_netG)  # (B, 6, H, W)
        
        # Render the inpainted output (using the optimized function!)
        pred_img = render_anisotropic_gaussians(param_out, mask, device=device)
        
        ###### 1. Update Discriminator ######
        optimizerD.zero_grad()
        # Real pass
        d_real = netD(gt_img)
        loss_d_real = adversarial_loss(d_real, is_real=True)
        # Fake pass
        d_fake = netD(pred_img.detach())
        loss_d_fake = adversarial_loss(d_fake, is_real=False)
        
        lossD = 0.5 * (loss_d_real + loss_d_fake)
        lossD.backward()
        optimizerD.step()
        
        ###### 2. Update Generator ######
        optimizerG.zero_grad()
        
        # Reconstruction loss
        loss_rec = reconstruction_loss(pred_img, gt_img, mask)
        
        # Smoothness loss on Cholesky factors
        ell_11 = param_out[:, 0:1, ...]
        ell_21 = param_out[:, 1:2, ...]
        ell_22 = param_out[:, 2:3, ...]
        loss_smooth = smoothness_loss(ell_11, ell_21, ell_22)
        
        # Adversarial
        d_fake_for_g = netD(pred_img)
        loss_adv = adversarial_loss(d_fake_for_g, is_real=True)
        
        # Total generator loss
        total_loss = lambda_rec * loss_rec \
                   + lambda_smooth * loss_smooth \
                   + lambda_adv * loss_adv
        
        total_loss.backward()
        optimizerG.step()
        
        # Record losses
        batch_losses_rec.append(loss_rec.item())
        batch_losses_smooth.append(loss_smooth.item())
        batch_losses_adv.append(loss_adv.item())
        batch_losses_d.append(lossD.item())
        
        # Update progress bar display
        loader_tqdm.set_postfix({
            'Rec': f"{loss_rec.item():.4f}",
            'Smooth': f"{loss_smooth.item():.4f}",
            'Adv': f"{loss_adv.item():.4f}",
            'D': f"{lossD.item():.4f}"
        })
    
    return batch_losses_rec, batch_losses_smooth, batch_losses_adv, batch_losses_d

def random_mask(img, hole_size=8):
    """
    Create a random square hole in a 3xHxW tensor image.
    Returns partial_img, mask.
    """
    _, h, w = img.shape
    top = torch.randint(0, h - hole_size, (1,)).item()
    left = torch.randint(0, w - hole_size, (1,)).item()
    
    mask = torch.ones((1, h, w), dtype=img.dtype)
    mask[:, top:top+hole_size, left:left+hole_size] = 0
    
    partial_img = img.clone()
    partial_img[:, top:top+hole_size, left:left+hole_size] = 0.0
    
    return partial_img, mask

class CIFAR10InpaintingDataset(Dataset):
    """
    Loads CIFAR-10 images and applies random_mask() to produce partial_img, mask, gt_img.
    """
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
        img, _ = self.dataset[idx]  # img: (3, 32, 32)
        partial_img, mask = random_mask(img, self.hole_size)
        gt_img = img
        return partial_img, mask, gt_img

def main_train_loop():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Networks
    netG = AnisotropicInpaintNet(in_channels=4, hidden_channels=64).to(device)
    netD = SimpleDiscriminator(in_channels=3).to(device)
    
    # Optimizers
    optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    # Dataloader: CIFAR-10 + random masking
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = CIFAR10InpaintingDataset(
        root='./data',
        train=True,
        transform=transform,
        hole_size=8  # adjust the size of the "hole"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )
    
    # Collect losses for plotting
    all_rec_losses = []
    all_smooth_losses = []
    all_adv_losses = []
    all_d_losses = []
    
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\n=== EPOCH {epoch+1}/{num_epochs} ===")
        rec_losses, smooth_losses, adv_losses, d_losses = train_epoch(
            netG, netD,
            train_loader,
            optimizerG, optimizerD,
            device=device
        )
        all_rec_losses.extend(rec_losses)
        all_smooth_losses.extend(smooth_losses)
        all_adv_losses.extend(adv_losses)
        all_d_losses.extend(d_losses)

    # Plot and save to file
    plt.figure(figsize=(8,6))
    plt.plot(all_rec_losses, label='Reconstruction Loss')
    plt.plot(all_smooth_losses, label='Smoothness Loss')
    plt.plot(all_adv_losses, label='Adversarial Loss')
    plt.plot(all_d_losses, label='Discriminator Loss')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.savefig("training_losses.png")
    plt.close()

    print("Training complete! Loss plots saved to 'training_losses.png'.")

if __name__ == "__main__":
    main_train_loop()

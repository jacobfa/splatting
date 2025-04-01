import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# For optional adversarial training
# pip install pytorch-lightning or similar if you like
# from torch.optim import Adam
# from pytorch_lightning import seed_everything, LightningModule, Trainer

def create_meshgrid(height, width, device='cpu'):
    """
    Returns a meshgrid of pixel coordinates: shape (2, H, W),
    where grid[0] is the x-coord, grid[1] is the y-coord.
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
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        
        # Final layer outputs 6 channels per pixel:
        #   (ell_11, ell_21, ell_22, c_r, c_g, c_b)
        self.conv_out = nn.Conv2d(hidden_channels, 6, kernel_size=1, padding=0)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (B, 4, H, W).
               Typically: [RGB_image concatenated with mask]
        Returns:
            param_out: (B, 6, H, W) tensor
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Raw outputs
        param_out = self.conv_out(x)
        return param_out


def render_anisotropic_gaussians(param_out, mask, device='cpu', truncate_thresh=1e-4):
    """
    Renders the final inpainted image by splatting anisotropic Gaussians.
    
    Args:
        param_out: (B, 6, H, W) from the network
                   [ell_11, ell_21, ell_22, c_r, c_g, c_b]
        mask:      (B, 1, H, W) binary mask (1=known, 0=hole)
        device:    torch device
        truncate_thresh: minimum weight threshold for inclusion
        
    Returns:
        inpainted: (B, 3, H, W) rendered image
    """
    B, _, H, W = param_out.shape
    
    # Separate out the predicted parameters
    ell_11 = param_out[:, 0:1, ...]  # shape (B,1,H,W)
    ell_21 = param_out[:, 1:2, ...]
    ell_22 = param_out[:, 2:3, ...]
    c_rgb  = param_out[:, 3:6, ...]  # shape (B,3,H,W)
    
    # Create coordinate grid: (2, H, W)
    grid = create_meshgrid(H, W, device=device)  # (2, H, W)
    
    # We'll accumulate numerator and denominator
    # in separate buffers of shape (B, 3, H, W)
    weighted_color_acc = torch.zeros((B, 3, H, W), device=device)
    weight_acc = torch.zeros((B, 1, H, W), device=device)
    
    # For each pixel i at (x_i, y_i), we have:
    #   L_i = [[ell_11, 0],
    #          [ell_21, ell_22]]
    #   Sigma_i = L_i * L_i^T
    # We invert Sigma_i for the Gaussian weight.

    # Convert grid to shape (2, H*W) for easier broadcasting
    xs_flat = grid[0].view(-1)  # (H*W,)
    ys_flat = grid[1].view(-1)
    
    # For each pixel, we interpret it as the center of the Gaussian
    # We'll gather the predicted parameters for that center.
    for b in range(B):
        # Flatten the parameter maps for batch b
        ell_11_b = ell_11[b].view(-1)  # (H*W,)
        ell_21_b = ell_21[b].view(-1)
        ell_22_b = ell_22[b].view(-1)
        c_rgb_b  = c_rgb[b].view(3, -1)  # (3, H*W)
        
        # We will loop over each pixel i in [0..H*W-1],
        # interpret it as the center of an anisotropic Gaussian.
        for i in range(H*W):
            # Center (x_i, y_i)
            x_i = xs_flat[i]
            y_i = ys_flat[i]
            
            # Build L_i
            L_11 = ell_11_b[i]
            L_21 = ell_21_b[i]
            L_22 = ell_22_b[i]
            
            # Construct Sigma_i^-1 analytically
            # Sigma = L L^T => inv(Sigma) = (L^T)^-1 * L^-1
            # For 2x2 lower-tri L, we can invert directly.
            # L = [[L_11, 0],
            #      [L_21, L_22]]
            # det(L) = L_11 * L_22
            # L^-1 = 1/det(L) * [[L_22, 0], [-L_21, L_11]]
            denom = L_11 * L_22
            # guard against zero or negative to keep stable
            eps = 1e-5
            denom = torch.clamp(denom, min=eps)
            
            L_inv_11 =  L_22 / denom
            L_inv_21 = -L_21 / denom
            L_inv_22 =  L_11 / denom
            
            # inv(Sigma) = L_inv^T @ L_inv
            # L_inv^T = [[L_inv_11, L_inv_21],
            #            [0,        L_inv_22]]
            
            # We'll define a small function to multiply the 2x2:
            # but for clarity, let's do it explicitly:
            # L_inv = [[L_inv_11, 0],
            #          [L_inv_21, L_inv_22]]
            
            # M = L_inv^T @ L_inv
            # shape (2x2)
            # First compute L_inv^T
            # L_inv^T = [[L_inv_11, L_inv_21],
            #            [0,        L_inv_22]]
            
            # Then multiply L_inv^T @ L_inv
            # = [[L_inv_11, L_inv_21],
            #    [0,        L_inv_22]]
            #   @
            #   [[L_inv_11, 0],
            #    [L_inv_21, L_inv_22]]
            
            M_00 = L_inv_11 * L_inv_11 + L_inv_21 * L_inv_21
            M_01 = L_inv_11 * 0 + L_inv_21 * L_inv_22
            M_10 = 0         * L_inv_11 + L_inv_22 * L_inv_21
            M_11 = 0         * 0 + L_inv_22 * L_inv_22
            
            # color for the i-th gaussian center
            c_i = c_rgb_b[:, i].unsqueeze(1)  # shape (3,1)
            
            # We'll compute the contribution for all pixels (x,y).
            # Let's re-construct the grid as shape (2, H, W).
            dx = grid[0] - x_i  # shape (H, W)
            dy = grid[1] - y_i  # shape (H, W)
            
            # Evaluate weight w_i(x,y) = exp( -1/2 [dx dy]^T * M * [dx dy] )
            # We do:
            # [dx dy] * M => [dx*M_00 + dy*M_10, dx*M_01 + dy*M_11]
            # Then dot that with [dx, dy] again.
            Wx = dx * M_00 + dy * M_10
            Wy = dx * M_01 + dy * M_11
            quad = Wx * dx + Wy * dy  # (H, W)
            
            w_i = torch.exp(-0.5 * quad)  # (H, W)
            
            # We can threshold small values
            w_i = torch.where(w_i < truncate_thresh, torch.zeros_like(w_i), w_i)
            
            # Accumulate
            weighted_color_acc[b, 0, ...] += w_i * c_i[0]
            weighted_color_acc[b, 1, ...] += w_i * c_i[1]
            weighted_color_acc[b, 2, ...] += w_i * c_i[2]
            
            weight_acc[b, 0, ...] += w_i
    
    # Final image = weighted sum / total weight
    # If weight is 0, we can default to input or 0 color.
    inpainted = torch.where(
        weight_acc > 1e-8,
        weighted_color_acc / (weight_acc + 1e-8),
        torch.zeros_like(weighted_color_acc)
    )
    
    return inpainted

def reconstruction_loss(pred, target, mask):
    """
    L1 or MSE loss on the known region only (where mask=1).
    """
    # pred, target: (B,3,H,W)
    # mask: (B,1,H,W)
    return F.l1_loss(pred * mask, target * mask)

def smoothness_loss(ell_11, ell_21, ell_22):
    """
    Encourages the Cholesky parameters to be smooth across adjacent pixels.
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
    
    netG.train()
    netD.train()
    
    for batch_idx, (partial_img, mask, gt_img) in enumerate(train_loader):
        partial_img = partial_img.to(device)  # (B,3,H,W)
        mask = mask.to(device)                # (B,1,H,W)
        gt_img = gt_img.to(device)            # (B,3,H,W)
        
        # Concatenate the mask as an extra channel
        input_netG = torch.cat([partial_img, mask], dim=1)  # (B,4,H,W)
        
        # Forward pass the generator
        param_out = netG(input_netG)  # (B,6,H,W)
        
        # Render the inpainted output
        pred_img = render_anisotropic_gaussians(param_out, mask, device=device)
        
        ###### 1. Update Discriminator ######
        optimizerD.zero_grad()
        # Real
        d_real = netD(gt_img)
        loss_d_real = adversarial_loss(d_real, is_real=True)
        
        # Fake
        d_fake = netD(pred_img.detach())
        loss_d_fake = adversarial_loss(d_fake, is_real=False)
        
        lossD = (loss_d_real + loss_d_fake) * 0.5
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
        
        total_loss = lambda_rec * loss_rec + \
                     lambda_smooth * loss_smooth + \
                     lambda_adv * loss_adv
        
        total_loss.backward()
        optimizerG.step()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(train_loader)} "
                  f"Rec: {loss_rec.item():.4f}, "
                  f"Smooth: {loss_smooth.item():.4f}, "
                  f"Adv: {loss_adv.item():.4f}, "
                  f"D: {lossD.item():.4f}")

def main_train_loop():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Example: instantiate the networks
    netG = AnisotropicInpaintNet(in_channels=4, hidden_channels=64).to(device)
    netD = SimpleDiscriminator(in_channels=3).to(device)
    
    # Create optimizer
    optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    # Suppose you have a DataLoader returning partial_img, mask, gt_img
    # train_loader = ...
    
    tra
    
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"=== EPOCH {epoch+1}/{num_epochs} ===")
        train_epoch(netG, netD, train_loader, optimizerG, optimizerD, device=device)
    
    # After training, netG can be used to inpaint new images.

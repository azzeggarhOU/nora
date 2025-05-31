# encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder for vMF SQ-VAE:
     - Maps input images x to continuous latents z_e.
     - Normalizes latent vectors along channel dim to lie on hypersphere S^{D-1}.
    """
    def __init__(self, in_channels=3, hidden_channels=128, embed_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, embed_dim,    kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        z_e = self.conv3(x)             # [B, D, H/4, W/4]
        # normalize to hypersphere S^{D-1} along channel axis
        B, D, H, W = z_e.shape
        z_flat = z_e.view(B, D, -1)     # [B, D, H*W]
        z_norm = F.normalize(z_flat, dim=1)
        z_e = z_norm.view(B, D, H, W)
        return z_e

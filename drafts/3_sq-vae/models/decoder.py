import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    Decoder for grayscale reconstruction:
     - Takes quantized latents z_q and reconstructs images with pixel values in [0,1].
    """
    def __init__(self, embed_dim=64, hidden_channels=128, out_channels=1):
        super().__init__()
        self.conv1   = nn.Conv2d(embed_dim,      hidden_channels, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_channels, out_channels,     kernel_size=4, stride=2, padding=1)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        # z_q: [B, D, H, W]
        x = F.relu(self.conv1(z_q))
        x = F.relu(self.deconv1(x))
        # Final activation ensures reconstructed pixels in [0,1]
        x_recon = torch.sigmoid(self.deconv2(x))
        return x_recon

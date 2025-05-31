# denoising_unit.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from denoise.dpm_plusplus_2_solver import DPMPlusPlus2Solver

class DenoisingUnit(nn.Module):
    def __init__(self, num_steps=1000, sigma_min=0.01, sigma_max=1.0):
        """
        Initialize the denoising unit.
        
        Args:
            num_steps (int): Number of timesteps for sampling.
            sigma_min (float): Minimum noise level.
            sigma_max (float): Maximum noise level.
        """
        super(DenoisingUnit, self).__init__()
        self.solver = DPMPlusPlus2Solver(num_steps, sigma_min, sigma_max)

    def forward(self, noisy_image, timesteps):
        """
        Apply denoising to a noisy image.
        
        Args:
            noisy_image (torch.Tensor): The noisy image to be denoised.
            timesteps (list of int): List of timesteps {t0, t1, ..., tM}.
        
        Returns:
            torch.Tensor: The denoised image.
        """
        return self.solver.denoise(noisy_image, timesteps)

    def visualize(self, noisy_image, denoised_image):
        """
        Visualize the noisy and denoised images side by side.
        
        Args:
            noisy_image (torch.Tensor): The noisy image.
            denoised_image (torch.Tensor): The denoised image.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(noisy_image.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Noisy Image")
        axes[0].axis("off")
        
        axes[1].imshow(denoised_image.permute(1, 2, 0).cpu().numpy())
        axes[1].set_title("Denoised Image")
        axes[1].axis("off")
        
        plt.show()

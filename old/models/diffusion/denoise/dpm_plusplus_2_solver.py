# dpmplusplus2_solver.py
import torch

class DPMPlusPlus2Solver:
    def __init__(self, num_steps, sigma_min, sigma_max, alpha_min=0.01, alpha_max=1.0, rho=7):
        """
        Initialize the DPM++2 solver with noise schedule and other parameters.
        
        Args:
            num_steps (int): Number of timesteps for sampling.
            sigma_min (float): Minimum noise level.
            sigma_max (float): Maximum noise level.
            alpha_min (float): Minimum alpha (controls step scaling).
            alpha_max (float): Maximum alpha (controls step scaling).
            rho (float): Controls the growth rate of the noise schedule.
        """
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.rho = rho
        self.sigmas = self._compute_sigmas()

    def _compute_sigmas(self):
        """
        Compute sigmas using the DPM++2 schedule with smooth scaling.

        Returns:
            torch.Tensor: Noise schedule values.
        """
        t = torch.linspace(0, 1, self.num_steps)
        
        # Noise scaling based on DPM++2 and Karras schedule
        sigmas = (self.sigma_max ** (1 / self.rho) + t * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        
        return sigmas

    def denoise(self, xT, timesteps):
        """
        Denoise the image using the DPM++2 solver.
        
        Args:
            xT (torch.Tensor): The noisy image at timestep 0.
            timesteps (list of int): List of timesteps {t0, t1, ..., tM}.
        
        Returns:
            torch.Tensor: The denoised image after processing.
        """
        M = len(timesteps)
        h = [0] * M  # Initialize h values
        x = xT.clone()  # Initial noisy state

        # Compute h values for each timestep
        for i in range(1, M):
            h[i] = self.sigmas[timesteps[i]] - self.sigmas[timesteps[i - 1]]

        Q_buffer = []  # Buffer to hold intermediate results

        # Initialize with the first timestep
        Q_buffer.append(x)
        
        # Loop through each timestep, applying the DPM++2 denoising algorithm
        for i in range(1, M):
            # Calculate the step sizes and differences
            ri = h[i - 1] / h[i] if h[i] != 0 else 1
            Di = 1 + 0.5 * ri

            # Update the value using the DPM++2 solver (denoising)
            x = x - (self.alpha_min * torch.exp(-h[i]) - 1) * Di

            # Add the result to the buffer
            Q_buffer.append(x)

        return Q_buffer[-1]

import torch
import matplotlib.pyplot as plt

class DPMPlusPlus2SolverInversion:
    def __init__(self, num_steps, sigma_min, sigma_max, alpha_min=0.01, alpha_max=1.0, rho=7):
        """
        Initialize the DPM++2 solver for inversion with noise schedule and other parameters.
        
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
        Compute sigmas using the DPM++2 schedule with smooth scaling for noise addition.

        Returns:
            torch.Tensor: Noise schedule values.
        """
        t = torch.linspace(0, 1, self.num_steps)
        
        # Noise scaling based on DPM++2 and Karras schedule (reversed for inversion)
        sigmas = (self.sigma_min ** (1 / self.rho) + t * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho
        
        return sigmas

    def solve(self, x0, timesteps):
        """
        Solve the DPM++2 inversion process by adding noise based on the schedule.
        
        Args:
            x0 (torch.Tensor): Initial value (clean image).
            timesteps (list of int): List of timesteps {t0, t1, ..., tM}.
        
        Returns:
            torch.Tensor: The result after adding noise.
        """
        M = len(timesteps)
        h = [0] * M  # Initialize h values
        x = x0.clone()  # Initial state

        # Compute h values for each timestep (this will now add noise)
        for i in range(1, M):
            h[i] = self.sigmas[timesteps[i]] - self.sigmas[timesteps[i - 1]]

        Q_buffer = []  # Buffer to hold intermediate results

        # Initialize with the clean image
        Q_buffer.append(x)
        
        # Loop through each timestep, applying the DPM++2 inversion process (noise addition)
        for i in range(1, M):
            # Calculate the step sizes and differences for noise addition
            ri = h[i] / h[i - 1] if h[i - 1] != 0 else 1
            Di = 1 + 0.5 * ri

            # Add noise to the image based on the inverse process
            x = x + (self.alpha_max * torch.exp(h[i]) - 1) * Di  # Adding noise (opposite of denoising)

            # Add the result to the buffer
            Q_buffer.append(x)

        return Q_buffer[-1]

# Example usage with image
if __name__ == "__main__":
    num_steps = 1000
    sigma_min = 0.01
    sigma_max = 1.0
    solver = DPMPlusPlus2SolverInversion(num_steps, sigma_min, sigma_max)

    # Create an initial clean image tensor (random for this example, [0, 1] normalized)
    x0 = torch.rand(1, 3, 64, 64)  # Example image (1x3x64x64 tensor)
    
    # Example timesteps for inversion (select timesteps to sample from)
    timesteps = list(range(100))  # Select timesteps to sample from

    # Solve the DPM++2 inversion process (adding noise)
    result = solver.solve(x0, timesteps)

    # Print the shape of the resulting noisy image
    print(result.shape)

    # Plot the calculated sigmas (noise schedule)
    plt.plot(solver.sigmas.numpy())
    plt.title("Noise Schedule (Sigmas) Over Time")
    plt.xlabel("Timesteps")
    plt.ylabel("Sigma Value")
    plt.grid(True)
    plt.show()

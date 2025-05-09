import torch
from forward.noise_scheduler import DPMPlusPlus2SolverInversion  # Assuming DPM++2Solver is in this file

def forward_process(image, time):
    """
    Adds noise to an image using a diffusion process with DPM++2Solver.

    Args:
        image: Input image (PyTorch tensor, normalized to [0, 1]).
        time: Current time step in the diffusion process (0 to num_steps-1).
        scheduler: DPM++2Solver instance with precomputed sigmas.

    Returns:
        image_noisy: Noisy image tensor.
        noise: The added noise tensor.
    """

    # Initialize the DPM++2Solver (with noise schedule)
    num_steps = 1000
    sigma_min = 0.01
    sigma_max = 1.0
    scheduler = DPMPlusPlus2SolverInversion(num_steps, sigma_min, sigma_max) 

    # Get the sigma for the current time step from the DPM++2Solver schedule
    sigma = scheduler.sigmas[time]
    
    # Standard Gaussian noise
    noise = torch.randn_like(image)
    
    # The DPM++2Solver method adjusts noise scaling here
    image_noisy = torch.sqrt(1 - sigma**2) * image + sigma * noise
    
    return image_noisy, noise

if __name__ == "__main__":
    # Example image tensor (normalized to [0, 1])
    image = torch.rand((1, 3, 64, 64))  # Replace with your actual image tensor

    # Perform the forward diffusion process for a specific time step
    time_step = 500  # Select any time step in the range [0, num_steps-1]
    image_noisy, noise = forward_process(image, time_step)

    # Print shapes as a sanity check
    print(f"Original Image Shape: {image.shape}")
    print(f"Noisy Image Shape: {image_noisy.shape}")
    print(f"Noise Shape: {noise.shape}")

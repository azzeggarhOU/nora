# main.py

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from forward.forward_diffusion import forward_process
from denoise.denoising_unit import DenoisingUnit  # Import DenoisingUnit

def load_image(image_path):
    """
    Loads an image and converts it to a PyTorch tensor.
    Args:
        image_path: Path to the input image.
    Returns:
        PyTorch tensor normalized to [0, 1].
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    return transform(image).float()

def resize_to_square(image, size):
    """
    Resizes a PyTorch tensor image to a square for visualization purposes.
    Args:
        image: PyTorch tensor of shape (C, H, W).
        size: Desired square size.
    Returns:
        Resized PIL image.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size))
    ])
    return transform(image)

def plot_images(images, steps, figure_size=(12, 9), save_path=None):
    """
    Plots square-resized images horizontally in a single row, ensuring the final plot is 4:3.
    Args:
        images: List of PyTorch tensors representing the images.
        steps: List of time steps corresponding to each image.
        figure_size: Tuple (width, height) to set the final diagram's aspect ratio.
        save_path: Optional path to save the plot.
    """
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=figure_size)
    
    for i, (img, step) in enumerate(zip(images, steps)):
        img_square = resize_to_square(img, 128)  # Resize each image to a square
        axes[i].imshow(img_square)
        axes[i].axis("off")
        axes[i].set_title(f"Step {step}", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def main():
    image_path = "test.jpg"  # Path to your test image
    time_steps_viz = [0, 200, 400, 600, 800, 1000]  # Time steps to visualize

    # Load the test image
    test_image = load_image(image_path)
    test_image = test_image.unsqueeze(0)  # Add batch dimension (1, C, H, W)

    # Initialize the Karras Scheduler
    time_steps = 1000

    # Generate noisy images for each time step
    noisy_images = []
    for t in range(time_steps):
        print("step " + str(t))
        noisy_image, _ = forward_process(test_image, t)
        if t in time_steps_viz:
            noisy_images.append(noisy_image.squeeze(0))  # Remove batch dimension for visualization

    # Instantiate the denoising unit
    denoising_unit = DenoisingUnit(num_steps=time_steps)

    # Denoise the noisy image at timestep 1000
    denoised_image = denoising_unit(noisy_images[-1], list(range(1000, 0)))  # Pass noisy image at t=1000 for denoising

    # Plot all images including the denoised image
    plot_images(noisy_images + [denoised_image.squeeze(0)], time_steps_viz + [1000], figure_size=(18, 6), save_path="noisy_and_denoised_images_diagram.png")

if __name__ == "__main__":
    main()

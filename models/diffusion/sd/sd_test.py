from diffusers import AutoencoderKL
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load pre-trained Stable Diffusion VAE
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2", subfolder="vae").to(device)

# Image preprocessing
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Adjust as needed
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

# Encode face and background into latent space
def encode_image(image):
    image_tensor = preprocess(image)
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
    return latents

# Decode latents back to image
def decode_latents(latents):
    with torch.no_grad():
        decoded_image = vae.decode(latents).sample
    decoded_image = ((decoded_image + 1) / 2).clamp(0, 1)  # Normalize to [0,1]
    return transforms.ToPILImage()(decoded_image.squeeze(0))

# Load images (face and background)
face_img = Image.open("face_crop.jpg").convert("RGB")
bg_img = Image.open("background.jpg").convert("RGB")

# Encode into latents
face_latents = encode_image(face_img)
bg_latents = encode_image(bg_img)

# Modify face latents (e.g., noise injection)
noise = torch.randn_like(face_latents) * 2  # Adjust noise level
face_latents_anon = face_latents + noise

# Decode anonymized face
anon_face_img = decode_latents(face_latents_anon)
anon_face_img.show()

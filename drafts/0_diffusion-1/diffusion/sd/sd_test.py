from diffusers import AutoencoderKL
import torch
from PIL import Image
import torchvision.transforms as transforms
import onnx

# Load pre-trained Stable Diffusion VAE
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2", subfolder="vae").to(device)

# Image preprocessing
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

# Encode image to latent space
def encode_image(image):
    image_tensor = preprocess(image)
    with torch.no_grad():
        dist = vae.encode(image_tensor).latent_dist
        latents = dist.mean + torch.randn_like(dist.mean) * dist.stddev  # Better sampling
    return latents

# Decode latent space back to image
def decode_latents(latents):
    with torch.no_grad():
        decoded_image = vae.decode(latents).sample
    decoded_image = ((decoded_image + 1) / 2).clamp(0, 1)
    return transforms.ToPILImage()(decoded_image.squeeze(0))

# Load images (face & background)
face_img = Image.open("cropped_face.jpg").convert("RGB")
bg_img = Image.open("background.jpg").convert("RGB")

# Encode images
face_latents = encode_image(face_img)
bg_latents = encode_image(bg_img)

# Modify face latents (Anonymization via noise)
noise = torch.randn_like(face_latents) * 2  # Adjust for stronger/noiser effect
face_latents_anon = face_latents + noise

# Decode and blend
anon_face_img = decode_latents(face_latents_anon)
blended_latents = 0.5 * face_latents_anon + 0.5 * bg_latents
blended_image = decode_latents(blended_latents)

# Save results
anon_face_img.save("anon_face.jpg")
blended_image.save("blended_face.jpg")

print("Anonymized face saved to 'anon_face.jpg'")
print("Blended image saved to 'blended_face.jpg'")

# Export VAE to ONNX
def export_vae_onnx(output_path="vae_encoder.onnx"):
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    torch.onnx.export(
        vae.encode,
        dummy_input,
        output_path,
        opset_version=11,
        input_names=["input"],
        output_names=["latent"],
    )
    print(f"VAE encoder exported to {output_path}")

export_vae_onnx()

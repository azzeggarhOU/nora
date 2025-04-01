import cv2
import torch
import numpy as np
from diffusers import AutoencoderKL

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Pretrained VAE Model using diffusers (using float32)
vae_model = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="vae"
).to(device)

# Define a wrapper that only uses the decoder part of the VAE.
class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super(VAEDecoderWrapper, self).__init__()
        self.vae = vae

    def forward(self, latent):
        latent_scaled = latent / 0.18215
        decoded = self.vae.decode(latent_scaled).sample
        return decoded

# Create an instance of the decoder wrapper
vae_decoder = VAEDecoderWrapper(vae_model).to(device)

# Preprocessing function for input image (resize, normalization)
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb

# 2. Face Detection (using OpenCV)
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        background = image.copy()
        background[y:y+h, x:x+w] = (0, 0, 0)  # Black out the face region
        return face, background, (x, y, w, h)
    return None, image, None

# 3. Latent Space Encoding using VAE (using the encoder part)
def encode_to_latent(image):
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
    image_tensor = image_tensor * 2 - 1
    with torch.no_grad():
        latent_dist = vae_model.encode(image_tensor).latent_dist
        latent = latent_dist.sample() * 0.18215
    return latent

# 4. Add Noise to the Latent Space
def add_noise_to_latent(latent):
    noise = torch.randn_like(latent) * 0.5
    noisy_latent = latent + noise
    return noisy_latent

# 5. Decode Latent to Image using VAE decoder wrapper
def decode_latent(noisy_latent):
    with torch.no_grad():
        decoded = vae_decoder(noisy_latent)
    decoded = (decoded + 1) / 2
    decoded = decoded.clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()[0] * 255
    decoded = decoded.astype(np.uint8)
    return decoded

# 6. Stitch Faces Back onto Background
def stitch_face_and_bg(original_bg, noisy_face, face_coords):
    (x, y, w, h) = face_coords
    noisy_face_resized = cv2.resize(noisy_face, (w, h))
    result_image = original_bg.copy()
    result_image[y:y+h, x:x+w] = noisy_face_resized
    return result_image

# Full process to handle an uploaded image
def process_image(image_path):
    img = preprocess_image(image_path)
    face, background, face_coords = detect_face(img)
    if face is None:
        print("No face detected")
        return None
    latent_face = encode_to_latent(face)
    noisy_latent_face = add_noise_to_latent(latent_face)
    noisy_face = decode_latent(noisy_latent_face)
    result_image = stitch_face_and_bg(background, noisy_face, face_coords)
    return result_image

image = process_image("test.jpeg")
if image is not None:
    cv2.imwrite("noisy_face.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
import os
import torch
from torch import nn
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt

# --- Configurations ---
DATA_DIR = './data/thumbnails64x64'
CHECKPOINT_PATH = './checkpoints/vqvae2_epoch_100.pth'
OUTPUT_DIR = './inference_outputs'
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model Definitions (matching training EMA buffers) ---
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_dim, 3, 1, 1)
        )
    def forward(self, x): return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh()
        )
    def forward(self, z): return self.deconv(z)

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, decay=0.99):
        super().__init__()
        self.embedding_dim    = embedding_dim
        self.num_embeddings   = num_embeddings
        self.embedding        = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w',            torch.randn(num_embeddings, embedding_dim))
        self.decay = decay
    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z_flat = z_e.permute(0,2,3,1).reshape(-1, C)
        dist = torch.cdist(z_flat, self.embedding)
        indices = torch.argmin(dist, dim=1)
        z_q = self.embedding[indices].view(B, H, W, C).permute(0,3,1,2)
        return z_q

# --- Instantiate models and load checkpoint ---
latent_dim = 128
num_embeddings = 512
encoder   = Encoder(latent_dim).to(DEVICE)
decoder   = Decoder(latent_dim).to(DEVICE)
quantizer = VectorQuantizerEMA(num_embeddings, latent_dim).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
quantizer.load_state_dict(checkpoint['quantizer'], strict=False)

encoder.eval(); decoder.eval(); quantizer.eval()

# --- Preprocessing and postprocessing ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
inv_norm = transforms.Compose([
    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    transforms.ToPILImage()
])

# --- Inference routine ---
def infer(image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    x   = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z_e = encoder(x)
        z_q = quantizer(z_e)
        recon = decoder(z_q)

    orig = inv_norm(x.squeeze().cpu())
    recon_img = inv_norm(recon.squeeze().cpu())

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(orig); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(recon_img); axes[1].set_title('Reconstruction'); axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- Run inference on sample images ---
if __name__ == '__main__':
    # collect all image file paths under DATA_DIR
    image_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in sorted(files):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(root, f))
    sample_images = image_files[:5]

    for in_path in sample_images:
        fname = os.path.basename(in_path)
        out_path = os.path.join(OUTPUT_DIR, f'recon_{fname}')
        infer(in_path, out_path)
        print(f'Saved reconstruction: {out_path}')

    print('Inference complete.')
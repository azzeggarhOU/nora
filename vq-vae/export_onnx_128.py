import os
import torch
from torch import nn
import torch.onnx

# --- Paths ---
CHECKPOINT = "./checkpoints/vqvae2_epoch_100.pth"  # Adjust as needed
ONNX_FILE  = "./vqvae2_128_web.onnx"
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LATENT_DIM = 128
NUM_CODEBOOK_ENTRIES = 512
EMA_DECAY = 0.99

# --- Model Definitions (from your training script) ---
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, LATENT_DIM, 3, 1, 1)
        )
    def forward(self, x):
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, z):
        return self.deconv(z)

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=NUM_CODEBOOK_ENTRIES, embedding_dim=LATENT_DIM, decay=EMA_DECAY):
        super().__init__()
        self.embedding_dim  = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding      = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.randn(num_embeddings, embedding_dim))
        self.decay = decay

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z_flat = z_e.permute(0,2,3,1).reshape(-1, C)
        idxs = []
        for i in range(0, z_flat.size(0), 1024):
            dist = torch.cdist(z_flat[i:i+1024], self.embedding)
            idxs.append(torch.argmin(dist, dim=1))
        indices = torch.cat(idxs)
        z_q = self.embedding[indices].view(B, H, W, C).permute(0,3,1,2)
        return z_q

# --- Full Model (wrapper) ---
class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = Encoder()
        self.quantizer = VectorQuantizerEMA()
        self.decoder   = Decoder()

    def forward(self, x):
        z_e = self.encoder(x)
        z_q = self.quantizer(z_e)
        return self.decoder(z_q)

# --- Load checkpoint ---
model = VQVAE().to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.encoder.load_state_dict(ckpt['encoder'])
model.decoder.load_state_dict(ckpt['decoder'])
model.quantizer.load_state_dict(ckpt['quantizer'], strict=False)  # EMA buffers relaxed
model.eval()

print("✅ Model loaded successfully.")

# --- Export to ONNX ---
dummy_input = torch.randn(1, 3, 128, 128, device=DEVICE)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_FILE,
    export_params=True,
    opset_version=20,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['reconstruction'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'reconstruction': {0: 'batch_size'}
    }
)

print(f"✅ Exported 128x128 ONNX model to {ONNX_FILE}")

import os
import torch
from torch import nn
import torch.onnx

# Paths
CHECKPOINT = "./checkpoints/vqvae2_epoch_100.pth"
ONNX_FILE  = "./vqvae2_web.onnx"
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Model Definitions (must match training) ---
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_dim, 3, 1, 1)
        )
    def forward(self, x):
        return self.conv(x)

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, decay=0.99):
        super().__init__()
        self.embedding_dim  = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding      = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.decay          = decay

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        # flatten to (N, C)
        z_flat = z_e.permute(0,2,3,1).reshape(-1, C)
        # compute squared L2 via broadcast: dist[n,m] = sum((z_flat[n]-emb[m])^2)
        # z_flat: [N,1,C], emb: [1,M,C]
        z_unsq = z_flat.unsqueeze(1)                 # [N,1,C]
        emb_unsq = self.embedding.unsqueeze(0)       # [1,M,C]
        diff = z_unsq - emb_unsq                     # [N,M,C]
        dist = (diff * diff).sum(-1)                 # [N,M]
        # find nearest embedding index
        indices = torch.argmin(dist, dim=1)          # [N] :contentReference[oaicite:4]{index=4}
        # gather embeddings
        z_q = self.embedding[indices]                \
               .view(B, H, W, C)                     \
               .permute(0,3,1,2)                     # [B,C,H,W]
        return z_q

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh()
        )
    def forward(self, z):
        return self.deconv(z)

# Wrapper that chains all modules
class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = Encoder(latent_dim=128)
        self.quantizer = VectorQuantizerEMA(num_embeddings=512, embedding_dim=128)
        self.decoder   = Decoder(latent_dim=128)
    def forward(self, x):
        z_e = self.encoder(x)
        z_q = self.quantizer(z_e)
        return self.decoder(z_q)

# --- Load checkpoint ---
model = VQVAE().to(DEVICE)                                  # :contentReference[oaicite:5]{index=5}
ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
model.encoder .load_state_dict(ckpt['encoder'])
model.decoder .load_state_dict(ckpt['decoder'])
# ignore EMA buffers
model.quantizer.load_state_dict(ckpt['quantizer'], strict=False)
model.eval()

# --- Export to ONNX ---
dummy = torch.randn(1, 3, 64, 64, device=DEVICE)          # :contentReference[oaicite:6]{index=6}

torch.onnx.export(
    model,                      # model to export
    dummy,                      # example input
    ONNX_FILE,                  # output path
    export_params=True,         # store weights inside the ONNX file
    opset_version=20,           # ORT Web supports opset 8+ (WebGPU) and 7–12, 13+ (WebNN) :contentReference[oaicite:7]{index=7}
    do_constant_folding=True,   # precompute constant nodes :contentReference[oaicite:8]{index=8}
    input_names=['input'],      # name of input node
    output_names=['reconstruction'],  # name of output node
    dynamic_axes={              # allow dynamic batch size
        'input':       {0: 'batch_size'},
        'reconstruction': {0: 'batch_size'}
    }
)
print(f"Exported ONNX model to {ONNX_FILE}")

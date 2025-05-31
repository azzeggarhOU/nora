import os
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models.encoder import Encoder
from models.decoder import Decoder
from models.quantizer import VmfQuantizer
from models.utils import log_train

# ------------------ Hyperparameters ------------------ #
BATCH_SIZE   = 128
EPOCHS       = 50
Z_DIM        = 6
NUM_CODES    = 512
LR           = 2e-4
SIGMA_RECON  = 0.1

# Precompute constants
RECON_COEFF = 1.0 / (2 * SIGMA_RECON ** 2)
CONST_TERM  = 0.5 * math.log(SIGMA_RECON ** 2)

# ------------------ Dataset ------------------ #
class dSpritesDataset(Dataset):
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)
        self.images  = data['imgs'].astype(np.float32)      # [N, H, W]
        self.latents = data['latents_values'].astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        x = np.expand_dims(x, axis=0)  # [1, H, W]
        return torch.from_numpy(x), torch.from_numpy(self.latents[idx])

# ------------------ Setup ------------------ #
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
ds     = dSpritesDataset('dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
dl     = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                    pin_memory=(device.type == 'cuda'))

encoder   = Encoder(in_channels=1, hidden_channels=128, embed_dim=Z_DIM).to(device)
decoder   = Decoder(embed_dim=Z_DIM, hidden_channels=128, out_channels=1).to(device)
quantizer = VmfQuantizer(num_embeddings=NUM_CODES, embedding_dim=Z_DIM).to(device)

optimizer = optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()) +
    list(quantizer.parameters()),
    lr=LR
)

writer = SummaryWriter(log_dir="runs/nora3")

# ------------------ Training Loop ------------------ #
global_step = 0
for epoch in range(1, EPOCHS + 1):
    encoder.train()
    decoder.train()
    quantizer.train()

    total_recon = 0.0
    total_reg   = 0.0

    for x, _ in tqdm(dl, desc=f"Epoch {epoch}/{EPOCHS}"):
        x = x.to(device)  # [B, 1, H, W]

        # Forward pass
        z_e = encoder(x)                           # [B, Z_DIM, H/4, W/4]
        z_q, reg, _ = quantizer(z_e)              # [B, Z_DIM, H/4, W/4], scalar, [B,H/4,W/4]
        x_hat = decoder(z_q)                       # [B, 1, H, W]

        # Compute loss
        recon = F.mse_loss(x_hat, x, reduction='mean') * RECON_COEFF
        loss  = recon + reg + CONST_TERM

        # Backprop & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_recon += recon.item()
        total_reg   += reg.item()

        # Logging
        if global_step % 100 == 0:
            writer.add_scalar("Loss/Reconstruction", recon.item(), global_step)
            writer.add_scalar("Loss/QuantizerReg",   reg.item(),   global_step)
            writer.add_scalar("Loss/Total",          loss.item(),  global_step)

            # Visualize reconstructions
            with torch.no_grad():
                grid = torch.cat([x[:8], x_hat[:8]], dim=0)
                writer.add_images("Reconstruction", grid, global_step)

            # Codebook usage histogram
            with torch.no_grad():
                B, C, Hq, Wq = z_e.shape
                flat_z = z_e.permute(0,2,3,1).reshape(-1, Z_DIM)       # [B*Hq*Wq, Z_DIM]
                embeddings = quantizer.embeddings.weight                # [K, Z_DIM]
                distances = torch.cdist(flat_z, embeddings)            # [N, K]
                indices   = distances.argmin(dim=1)                    # [N]
                usage_hist = torch.bincount(indices, minlength=NUM_CODES).float()
                writer.add_histogram("Codebook/Usage", usage_hist, global_step)

        global_step += 1

    # End of epoch logging
    avg_recon = total_recon / len(dl)
    avg_reg   = total_reg   / len(dl)
    log_train(epoch, {"Recon": avg_recon, "QuantReg": avg_reg})

# Save model checkpoints
os.makedirs("checkpoints", exist_ok=True)
torch.save({
    "encoder":   encoder.state_dict(),
    "decoder":   decoder.state_dict(),
    "quantizer": quantizer.state_dict()
}, "checkpoints/sqvae.pth")

writer.close()

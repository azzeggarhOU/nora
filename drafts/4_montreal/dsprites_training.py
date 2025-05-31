#!/usr/bin/env python3
import os
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm
import torchvision.utils as vutils
from torch.amp import autocast, GradScaler
import gc
import numpy as np

CHECKPOINT_DIR = "checkpoints"

# --------------------------------------------------
# 1) Dataset
# --------------------------------------------------
class dSpritesDataset(Dataset):
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)
        # original images: shape (N, H, W)
        self.images  = data['imgs'].astype(np.float32)
        self.latents = data['latents_values'].astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x_gray = self.images[idx]                    # shape: (H, W)
        # stack into 3 identical channels:
        x_rgb  = np.stack([x_gray, x_gray, x_gray], axis=0)  # shape: (3, H, W)
        u      = self.latents[idx]
        return torch.from_numpy(x_rgb), torch.from_numpy(u)


# --------------------------------------------------
# 2) Encoder (reduced for 64x64)
# --------------------------------------------------
class CausalEncoder(nn.Module):
    def __init__(self, latent_dim=6, u_dim=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), # 128 → 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 64 → 32
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_dim, 3, 1, 1) # 32 → 32 (just change channels)

        )
    def forward(self, x, u):
        return self.conv(x)

# --------------------------------------------------
# 3) Vector Quantizer EMA
# --------------------------------------------------
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99):
        super().__init__()
        self.embedding_dim  = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding      = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w',            torch.randn(num_embeddings, embedding_dim))
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

        if self.training:
            encodings = torch.zeros(z_flat.size(0), self.num_embeddings, device=z_flat.device)
            encodings.scatter_(1, indices.unsqueeze(1), 1)
            n = encodings.sum(0)
            dw = encodings.t() @ z_flat
            # EMA update on CPU to reduce MPS memory
            ema_cluster = self.ema_cluster_size.cpu().mul(self.decay).add(n.cpu(), alpha=1-self.decay)
            ema_w_buf    = self.ema_w.cpu().mul(self.decay).add(dw.cpu(), alpha=1-self.decay)
            total = ema_cluster.sum()
            eps = torch.tensor(1e-5, device=ema_cluster.device, dtype=ema_cluster.dtype)
            cluster_size = ema_cluster.add(eps).div(total + self.num_embeddings*1e-5).mul(total)
            new_emb = ema_w_buf.div(cluster_size.unsqueeze(1))
            # copy back
            self.ema_cluster_size.copy_(ema_cluster.to(self.ema_cluster_size.device))
            self.ema_w.copy_(ema_w_buf.to(self.ema_w.device))
            self.embedding.data.copy_(new_emb.to(self.embedding.device))
        return z_q

# --------------------------------------------------
# 4) Decoder (mirrored, now with tanh)
# --------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim=6):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1), # 32 → 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 64 → 128
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()

        )

    def forward(self, z):
        return self.deconv(z)

# --------------------------------------------------
# 5) CausalVQ-VAE baseline
# --------------------------------------------------
class CausalVQVAE(nn.Module):
    def __init__(self, latent_dim=6, num_codes=512):
        super().__init__()
        self.encoder       = CausalEncoder(latent_dim, latent_dim)
        self.quantizer     = VectorQuantizerEMA(num_codes, latent_dim)
        self.decoder       = Decoder(latent_dim)

    def forward(self, x, u):
        z_e = self.encoder(x, u)
        z_q = self.quantizer(z_e)
        xr  = self.decoder(z_q)
        return xr, z_e, z_q

# --------------------------------------------------
# 6) Training Loop using MSE (scaled for tanh)
# --------------------------------------------------
def train():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    ds = dSpritesDataset('dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    dl = DataLoader(ds, batch_size=128, shuffle=True, pin_memory=True)
    model = CausalVQVAE(latent_dim=6, num_codes=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    writer = SummaryWriter('runs/causal_vqvae_baseline_7')
    scaler = GradScaler(enabled=(device.type == "cuda"))

    val_x, val_u = next(iter(dl))
    val_x, val_u = val_x.to(device), val_u.to(device)

    start_epoch = 1
    epochs = 80
    for ep in range(start_epoch, epochs+1):
        model.train()
        total_loss = 0.0
        for bi, (x, u) in enumerate(tqdm(dl, desc=f"Epoch {ep}")):
            x, u = x.to(device), u.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="mps", dtype=torch.float16, enabled=(device.type!="cpu")):
                xr, z_e, z_q = model(x, u)
                recon_loss = torch.mean((x - xr) ** 2)
                commit_loss = 0.25 * torch.mean((z_e.detach() - z_q) ** 2)
                loss = recon_loss + commit_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if bi % 100 == 0:
                model.eval()
                with torch.no_grad():
                    val_imgs = val_x[:8]
                    recon_val, _, _ = model(val_imgs, val_u[:8])  # single call for validation
                    grid = utils.make_grid(torch.cat([val_imgs, recon_val], dim=0), nrow=val_imgs.size(0))
                    writer.add_image('Validation/Reconstruction', grid, ep * len(dl) + bi)
                    writer.add_scalar('batch/loss', loss.item(), ep * len(dl) + bi)

                del val_imgs, recon_val, grid
                gc.collect()
                if device.type == 'mps': torch.mps.empty_cache()

            del x, z_e, z_q, xr, loss, recon_loss, commit_loss
            gc.collect()
            if device.type == 'mps': torch.mps.empty_cache()

        avg_loss = total_loss / len(dl)
        print(f"[Epoch {ep}] Loss = {avg_loss:.4f}")
        writer.add_scalar('epoch/loss', avg_loss, ep)
        scheduler.step()

        if ep % 10 == 0:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict(),
                'quantizer': model.quantizer.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(CHECKPOINT_DIR, f"vqcausalvae_epoch_{ep}.pth"))

    writer.close()

if __name__ == '__main__':
    train()

#!/usr/bin/env python3
"""
Training script for Stochastic VQ-VAE 128x128.

- Encoder outputs a distribution (mu, logvar)
- Reparameterization trick is used to sample z ~ N(mu, sigma^2)
- KL divergence loss added
- Checkpoints every 5 epochs
- Exits cleanly after checkpoint
- Supports auto-resume
"""

import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import gc

# --- Configurations ---
DATA_DIR = "./data/thumbnails128x128"
BATCH_SIZE = 64
LATENT_DIM = 128
NUM_CODEBOOK_ENTRIES = 512
EMA_DECAY = 0.99
LEARNING_RATE = 2e-4
EPOCHS = 100
BETA = 0.25
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "runs/vqvae2_mps"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# --- Data ---
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_dataset.samples = train_dataset.samples[:10000]

val_dataset = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)
val_dataset.samples = val_dataset.samples[:10000]

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, persistent_workers=False, drop_last=True
)

val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False,
    num_workers=0, persistent_workers=False, drop_last=True
)

# --- Model Components ---
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
        self.mu_head = nn.Conv2d(LATENT_DIM, LATENT_DIM, 1)
        self.logvar_head = nn.Conv2d(LATENT_DIM, LATENT_DIM, 1)

    def forward(self, x):
        x = self.conv(x)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar

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
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
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

def reparameterize(mu, logvar):
    """Reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# --- Model setup ---
encoder = Encoder().to(device)
decoder = Decoder().to(device)
quantizer = VectorQuantizerEMA().to(device)

optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()) + list(quantizer.parameters()), lr=LEARNING_RATE
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
writer = SummaryWriter(LOG_DIR)
scaler = GradScaler(enabled=(device.type == "cuda"))

# --- Resume from latest checkpoint ---
start_epoch = 1
if os.listdir(CHECKPOINT_DIR):
    checkpoints = sorted([
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.endswith('.pth')
    ], key=lambda x: int(x.split('_')[-1].replace('.pth', '')))

    if checkpoints:
        latest_ckpt = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
        print(f"Loading checkpoint {latest_ckpt}...")
        checkpoint_data = torch.load(latest_ckpt, map_location=device)
        encoder.load_state_dict(checkpoint_data['encoder'])
        decoder.load_state_dict(checkpoint_data['decoder'])
        quantizer.load_state_dict(checkpoint_data['quantizer'])
        optimizer.load_state_dict(checkpoint_data['optimizer'])
        start_epoch = int(latest_ckpt.split('_')[-1].replace('.pth', '')) + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No valid checkpoint found, starting from scratch.")

# --- Training loop ---
if __name__ == "__main__":
    for epoch in range(start_epoch, EPOCHS+1):
        encoder.train(); decoder.train(); quantizer.train()
        total_loss = 0.0

        for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="mps", dtype=torch.float16, enabled=(device.type != "cpu")):
                mu, logvar = checkpoint(encoder, imgs, use_reentrant=False)
                z = reparameterize(mu, logvar)
                z_q = quantizer(z)
                recon = decoder(z_q)
                recon_loss = torch.mean((imgs - recon) ** 2)
                commit_loss = 0.25 * torch.mean((z.detach() - z_q) ** 2)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + commit_loss + BETA * kl_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            del imgs, mu, logvar, z, z_q, recon, loss, recon_loss, commit_loss, kl_loss
            gc.collect()
            torch.mps.empty_cache()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        scheduler.step()

        # --- Validation every 5 epochs ---
        if epoch % 5 == 0:
            encoder.eval(); decoder.eval(); quantizer.eval()
            with torch.no_grad():
                val_imgs, _ = next(iter(val_loader))
                val_imgs = val_imgs.to(device, non_blocking=True)
                with autocast(device_type="mps", dtype=torch.float16, enabled=(device.type != "cpu")):
                    mu_val, logvar_val = encoder(val_imgs)
                    z_val = reparameterize(mu_val, logvar_val)
                    z_q_val = quantizer(z_val)
                    recon_val = decoder(z_q_val)
                grid = utils.make_grid(torch.cat([val_imgs, recon_val], dim=0), nrow=val_imgs.size(0))
                writer.add_image('Validation/Reconstruction', grid, epoch)
                del val_imgs, mu_val, logvar_val, z_val, z_q_val, recon_val, grid
                gc.collect()
                torch.mps.empty_cache()

        # --- Save checkpoint and exit every 5 epochs ---
        if epoch % 5 == 0 or epoch == EPOCHS:
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'quantizer': quantizer.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(CHECKPOINT_DIR, f"vqvae2_epoch_{epoch}.pth"))
            print(f"\n✅ Epoch {epoch} completed and checkpoint saved. Exiting. Please restart the script to continue training.")
            writer.close()
            sys.exit(0)

    writer.close()

import os
# Adjust MPS allocator to release memory aggressively
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

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
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "runs/vqvae2_mps"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# --- Data Transforms & Loaders ---
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
raw_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])



train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
val_dataset   = datasets.ImageFolder(root=DATA_DIR, transform=raw_transform)

train_dataset.samples = train_dataset.samples[:10000]
val_dataset.samples   = val_dataset.samples[:10000]

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    persistent_workers=False,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    persistent_workers=False,
    drop_last=True
)

# --- Model Components ---
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), # 128 → 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 64 → 32
            nn.ReLU(inplace=True),
            nn.Conv2d(128, LATENT_DIM, 3, 1, 1) # 32 → 32 (just change channels)

        )
    def forward(self, x):
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 128, 4, 2, 1), # 32 → 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 64 → 128
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()

        )
    def forward(self, z):
        return self.deconv(z)

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay):
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

# instantiate models
encoder   = Encoder().to(device)
decoder   = Decoder().to(device)
quantizer = VectorQuantizerEMA(NUM_CODEBOOK_ENTRIES, LATENT_DIM, EMA_DECAY).to(device)

optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()) + list(quantizer.parameters()),
    lr=LEARNING_RATE
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
writer    = SummaryWriter(LOG_DIR)
# Use scaler only on CUDA (disabled on MPS)
scaler    = GradScaler(enabled=(device.type == "cuda"))

# --- Training Loop ---
if __name__ == "__main__":
    start_epoch = 1 

    if os.listdir(CHECKPOINT_DIR):
        checkpoints = sorted(os.listdir(CHECKPOINT_DIR))
        latest_ckpt = os.path.join(CHECKPOINT_DIR, checkpoints[-1])

        if latest_ckpt:
            print(f"Loading checkpoint {latest_ckpt}...")
            checkpoint_data = torch.load(latest_ckpt, map_location=device)
            encoder.load_state_dict(checkpoint_data['encoder'])
            decoder.load_state_dict(checkpoint_data['decoder'])
            quantizer.load_state_dict(checkpoint_data['quantizer'])
            optimizer.load_state_dict(checkpoint_data['optimizer'])
            # You can also parse epoch number from filename if you want
            start_epoch = int(latest_ckpt.split('_')[-1].replace('.pth', '')) + 1
        else:
            start_epoch = 1


    for epoch in range(start_epoch, EPOCHS+1):
        encoder.train(); decoder.train(); quantizer.train()
        total_loss = 0.0
        for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="mps", dtype=torch.float16, enabled=(device.type!="cpu")):
                z_e = checkpoint(encoder, imgs, use_reentrant=False)
                z_q = quantizer(z_e)
                recon = decoder(z_q)
                recon_loss = torch.mean((imgs - recon) ** 2)
                commit_loss = 0.25 * torch.mean((z_e.detach() - z_q) ** 2)
                loss = recon_loss + commit_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            del imgs, z_e, z_q, recon, loss, recon_loss, commit_loss
            gc.collect()
            torch.mps.empty_cache()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        scheduler.step()

        # validation every 5 epochs
        if epoch % 5 == 0:
            encoder.eval(); decoder.eval(); quantizer.eval()
            with torch.no_grad():
                val_imgs, _ = next(iter(val_loader))
                val_imgs = val_imgs.to(device, non_blocking=True)
                with autocast(device_type="mps", dtype=torch.float16, enabled=(device.type!="cpu")):
                    z_e_val = encoder(val_imgs)
                    z_q_val = quantizer(z_e_val)
                    recon_val = decoder(z_q_val)
                grid = utils.make_grid(torch.cat([val_imgs, recon_val], dim=0), nrow=val_imgs.size(0))
                writer.add_image('Validation/Reconstruction', grid, epoch)
                del val_imgs, z_e_val, z_q_val, recon_val, grid
                gc.collect()
                torch.mps.empty_cache()

        # checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'encoder'  : encoder.state_dict(),
                'decoder'  : decoder.state_dict(),
                'quantizer': quantizer.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(CHECKPOINT_DIR, f"vqvae2_epoch_{epoch}.pth"))

    writer.close()

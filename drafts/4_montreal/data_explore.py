import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# --- Dataset with weak supervision labels u ---
class dSpritesDataset(Dataset):
    """
    Loads dSprites images and latent factor labels (Section 3.1, Eq. (7))
    """
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)
        self.images = data['imgs'].astype(np.float32)      # (N,64,64)
        self.latents = data['latents_values'].astype(np.float32)  # (N,6)
        self.latent_dim = self.latents.shape[1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx][None, :, :]  # add channel dim
        u = self.latents[idx]             # weak supervision labels
        return torch.from_numpy(x), torch.from_numpy(u)

# --- Encoder q(eps|x) = N(mu(x), σ²(x))  (Section 3, q(ε|x)) ---
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Convolutional encoder (improves reconstruction)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),    # 64→32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),   # 32→16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),  # 16→8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(), # 8→4
        )
        self.flatten = nn.Flatten()
        self.fc_mu_logvar = nn.Linear(256 * 4 * 4, latent_dim * 2)

    def forward(self, x):
        h = self.conv(x)
        h = self.flatten(h)
        stats = self.fc_mu_logvar(h)
        mu, logvar = stats.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = mu + std * torch.randn_like(std)     # sample ε ~ q(ε|x)
        # KL(q(ε|x)||p(ε)=N(0,I))  (Section 4.1, ELBO)
        kl_eps = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return eps, mu, logvar, kl_eps

# --- Structural Causal Model Layer z = (I - Aᵀ)⁻¹ ε  (Section 3.1, Eq. (1)) ---
class CausalLayer(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.A = nn.Parameter(torch.zeros(latent_dim, latent_dim))

    def forward(self, eps):
        eye = torch.eye(self.latent_dim, device=eps.device)
        M = eye - self.A.T
        if M.device.type == 'mps':
            M_cpu, eps_cpu = M.cpu(), eps.cpu()
            z_cpu = torch.linalg.solve(M_cpu, eps_cpu.T).T
            return z_cpu.to(eps.device)
        return torch.linalg.solve(M, eps.T).T

    def dag_penalty(self):
        A_sq = self.A * self.A
        if A_sq.device.type == 'mps':
            A_sq_cpu = A_sq.detach().cpu()
            expm_cpu = torch.matrix_exp(A_sq_cpu)
            return torch.trace(expm_cpu) - self.latent_dim
        expm = torch.matrix_exp(A_sq)
        return torch.trace(expm) - self.latent_dim

# --- Decoder p(x|z) ---
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Mirror of encoder
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),  # 4→8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),   # 8→16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),    # 16→32
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()   # 32→64
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, 4, 4)
        return self.deconv(h)

# --- Training Setup ---
dataset = dSpritesDataset("dataset/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
latent_dim = dataset.latent_dim  # 6 factors (Section 3.1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
batch_size, num_epochs, lr = 128, 100, 1e-4
lambda_dag = 1.0

# Data and writer
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
writer = SummaryWriter("runs/causalvae_conv")

# Models
encoder = Encoder(latent_dim).to(device)
causal = CausalLayer(latent_dim).to(device)
decoder = Decoder(latent_dim).to(device)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(causal.parameters()) + list(decoder.parameters()), lr=lr
)

# --- Training Loop ---
for epoch in range(num_epochs):
    running = {"recon":0, "kl_eps":0, "kl_z":0, "dag":0}
    for x, u in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        x,u = x.to(device), u.to(device)
        eps, mu, logvar, kl_eps = encoder(x)
        z = causal(eps)

        # Conditional KL (Section 4.1, Eq. 10)
        I = torch.eye(latent_dim, device=device)
        K = torch.linalg.inv(I - causal.A.T)
        mu_z = mu @ K.T
        var_z = (logvar.exp()) @ (K**2)
        std_z = torch.sqrt(var_z + 1e-8)
        kl_z = 0.5 * torch.mean(torch.sum(var_z + (mu_z - u)**2 - 1 - 2*torch.log(std_z + 1e-8), dim=1))

        x_hat = decoder(z)
        recon = F.binary_cross_entropy(x_hat, x, reduction='mean')
        dag = causal.dag_penalty()
        loss = recon + kl_eps + kl_z + lambda_dag * dag

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(causal.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad(): causal.A.clamp_(-0.1, 0.1)

        running["recon"] += recon.item()
        running["kl_eps"] += kl_eps.item()
        running["kl_z"] += kl_z.item()
        running["dag"] += dag.item()

    # Log averages
    for k,v in running.items(): writer.add_scalar(k, v/len(dataloader), epoch)

    # Visualize every 5 epochs
    if (epoch+1)%5==0:
        with torch.no_grad():
            x0, _ = next(iter(dataloader))
            x0 = x0.to(device)
            eps0,_ ,_,_ = encoder(x0)
            recon0 = decoder(causal(eps0))
            writer.add_images("orig", x0, epoch)
            writer.add_images("recon", recon0, epoch)

# Done
torch.cuda.empty_cache(); writer.close()

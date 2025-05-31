# quantizer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class VmfQuantizer(nn.Module):
    """
    vMF‐based Quantizer for SQ‐VAE:
     - Maintains a codebook of K embeddings on S^{D-1}.
     - Computes trainable von Mises–Fisher posterior:
         P̂ϕ(zq_i = b_k | z_i) = softmax_k(κϕ · ⟨b_k, z_i⟩)  (Eq. 13) :contentReference[oaicite:2]{index=2}
     - Samples discrete codes, returns z_q and the vMF regularization:
         Rᵥᴹᶠ = ∑ᵢ κϕ·(1 − ⟨z_q,i, z_i⟩)  (Eq. 25) :contentReference[oaicite:3]{index=3}.
    """
    def __init__(self, num_embeddings=512, embedding_dim=64, init_kappa=1.0):
        super().__init__()
        self.embedding_dim   = embedding_dim
        self.num_embeddings  = num_embeddings
        self.kappa_phi       = nn.Parameter(torch.tensor(init_kappa))
        self.embeddings      = nn.Embedding(num_embeddings, embedding_dim)
        # initialize codebook uniformly on sphere
        nn.init.uniform_(self.embeddings.weight, -1.0, 1.0)
        with torch.no_grad():
            self.embeddings.weight.data = F.normalize(self.embeddings.weight.data, dim=1)

    def forward(self, z_e: torch.Tensor):
        """
        Args:
          z_e: continuous latents [B, D, H, W] already normalized on S^{D-1}.
        Returns:
          z_q: quantized latents [B, D, H, W]
          reg: vMF regularization scalar
          indices: discrete code indices [B, H, W]
        """
        B, D, H, W = z_e.shape
        # flatten spatial dims
        z_perm  = z_e.permute(0,2,3,1).contiguous()  # [B, H, W, D]
        flat_z  = z_perm.view(-1, D)                 # [N, D], N = B*H*W

        # normalize codebook on sphere
        emb_w   = F.normalize(self.embeddings.weight, dim=1)  # [K, D]

        # cosine similarities and posterior logits
        sims    = flat_z @ emb_w.t()                   # [N, K]
        logits  = self.kappa_phi * sims                # trainable concentration
        probs   = F.softmax(logits, dim=1)

        # sample discrete codes
        indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # [N]
        z_q_flat = emb_w[indices]                                    # [N, D]

        # reshape back to [B, D, H, W]
        z_q = z_q_flat.view(B, H, W, D).permute(0,3,1,2).contiguous()

        # vMF regularization: Rᵥᴹᶠ = ∑ κϕ (1 − cos(z_q, z_e))
        chosen_sims = sims[torch.arange(sims.size(0)), indices]
        reg = torch.mean(self.kappa_phi * (1 - chosen_sims))

        return z_q, reg, indices.view(B, H, W)

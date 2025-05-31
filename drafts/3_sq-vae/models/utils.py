import torch

def reconstruction_loss(x_hat, x):
    return torch.nn.functional.mse_loss(x_hat, x)

def log_train(epoch, loss_dict):
    print(f"[Epoch {epoch}] " + " | ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items()))

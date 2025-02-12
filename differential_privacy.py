import torch

def add_noise(tensor, noise_scale):
    """Add Gaussian noise to tensor for differential privacy"""
    if noise_scale == 0:
        return tensor
    
    noise = torch.randn_like(tensor) * noise_scale
    return tensor + noise

def compute_sensitivity(tensor):
    """Compute L2 sensitivity of tensor"""
    return torch.norm(tensor)

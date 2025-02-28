"""
Differential privacy utilities for federated learning.
"""

import numpy as np
import torch
#from src.utils.database import get_db #This import is commented out because it's not used and causes an error if the database module doesn't exist.


def add_noise(tensor, noise_scale):
    """
    Add Gaussian noise to a tensor for differential privacy.

    Args:
        tensor: PyTorch tensor to add noise to
        noise_scale: Standard deviation of Gaussian noise

    Returns:
        Tensor with added noise
    """
    if noise_scale <= 0:
        return tensor

    # Generate noise with same shape as tensor
    noise = torch.randn_like(tensor) * noise_scale

    # Add noise to tensor
    return tensor + noise

def calculate_privacy_loss(noise_scale, num_selected, total_clients):
    """
    Calculate the privacy loss (epsilon) for a federated round.

    Args:
        noise_scale: Standard deviation of Gaussian noise
        num_selected: Number of clients selected in this round
        total_clients: Total number of clients

    Returns:
        Privacy loss (epsilon)
    """
    # Simple privacy accounting model - in practice more sophisticated
    # accounting methods like RDP or zCDP would be used
    if noise_scale <= 0:
        return float('inf')

    # Participation ratio affects privacy loss
    participation_ratio = num_selected / total_clients

    # More noise means less privacy loss
    privacy_loss = participation_ratio / (noise_scale ** 2)

    return privacy_loss
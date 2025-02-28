import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.privacy.differential_privacy import add_noise, compute_sensitivity

def test_add_noise():
    """Test adding Gaussian noise for differential privacy"""
    # Create test tensor
    test_tensor = torch.ones(100, 100)
    
    # Test with zero noise scale (should return original tensor)
    noisy_tensor_zero = add_noise(test_tensor, 0)
    assert torch.allclose(noisy_tensor_zero, test_tensor), "Zero noise scale should return original tensor"
    
    # Test with small noise scale
    small_noise_scale = 0.01
    noisy_tensor_small = add_noise(test_tensor, small_noise_scale)
    
    # Check that noise was added (tensor should be different)
    assert not torch.allclose(noisy_tensor_small, test_tensor), "Noise should be added to tensor"
    
    # Check that the mean of the noise is approximately zero
    # For Gaussian noise with mean zero, the noisy tensor should have mean close to the original
    assert abs(noisy_tensor_small.mean().item() - test_tensor.mean().item()) < 0.05, \
        "Mean of noisy tensor should be close to original"
    
    # Test with larger noise scale
    large_noise_scale = 0.1
    noisy_tensor_large = add_noise(test_tensor, large_noise_scale)
    
    # Check that larger noise scale results in larger deviation
    small_deviation = (noisy_tensor_small - test_tensor).abs().mean().item()
    large_deviation = (noisy_tensor_large - test_tensor).abs().mean().item()
    assert large_deviation > small_deviation, "Larger noise scale should result in larger deviation"
    
    print("Differential privacy noise addition test passed!")

def test_privacy_loss_calculation():
    """Test privacy loss calculation in federated learning"""
    # Import the FederatedLearning class
    from federated_learning import FederatedLearning
    from model import SimpleConvNet
    
    # Create a simple model and federated learning instance
    model = SimpleConvNet()
    fl = FederatedLearning(model, num_clients=3)
    
    # Test with different noise scales and query numbers
    # Higher noise should give lower privacy loss
    privacy_budget = 1.0
    high_noise = 1.0
    low_noise = 0.1
    
    # Single query
    high_noise_loss = fl.calculate_privacy_loss(privacy_budget, high_noise, num_queries=1)
    low_noise_loss = fl.calculate_privacy_loss(privacy_budget, low_noise, num_queries=1)
    
    assert high_noise_loss < low_noise_loss, "Higher noise should result in lower privacy loss"
    
    # Multiple queries should increase privacy loss
    multi_query_loss = fl.calculate_privacy_loss(privacy_budget, high_noise, num_queries=10)
    assert multi_query_loss > high_noise_loss, "Multiple queries should increase privacy loss"
    
    # Zero noise should result in infinite privacy loss
    zero_noise_loss = fl.calculate_privacy_loss(privacy_budget, 0, num_queries=1)
    assert zero_noise_loss == float('inf'), "Zero noise should result in infinite privacy loss"
    
    print("Privacy loss calculation test passed!")

if __name__ == "__main__":
    print("Testing differential privacy mechanisms...")
    test_add_noise()
    test_privacy_loss_calculation()
    print("\nAll differential privacy tests passed!")
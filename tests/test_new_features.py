
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from data_handler import load_mnist_data, create_non_iid_data_indices, distribute_data_non_iid
from model import SimpleConvNet
from federated_learning import FederatedLearning

def test_client_selection():
    """Test client selection for partial participation"""
    model = SimpleConvNet()
    fl = FederatedLearning(model, num_clients=10)
    
    # Test with different fractions
    for fraction in [0.3, 0.5, 0.8]:
        selected = fl.select_clients(fraction=fraction)
        assert len(selected) == int(10 * fraction) or len(selected) == int(10 * fraction) + 1, \
            f"Expected ~{int(10 * fraction)} clients, got {len(selected)}"
    
    # Test with minimum clients
    selected = fl.select_clients(fraction=0.1, min_clients=3)
    assert len(selected) >= 3, f"Expected at least 3 clients, got {len(selected)}"
    
    print("Client selection test passed!")

def test_model_compression():
    """Test model compression functionality"""
    model = SimpleConvNet()
    fl = FederatedLearning(model, num_clients=3)
    
    # Create a sample state dictionary
    state_dict = model.state_dict()
    
    # Test compression with different ratios
    for ratio in [0.1, 0.5, 0.9]:
        compressed = fl.compress_model(state_dict, compression_ratio=ratio)
        
        # Check that all keys are preserved
        assert set(compressed.keys()) == set(state_dict.keys()), "Keys don't match after compression"
        
        # For some layer, count non-zero values to verify compression
        for key in state_dict.keys():
            if isinstance(state_dict[key], torch.Tensor) and state_dict[key].numel() > 10:
                original_nonzero = torch.count_nonzero(state_dict[key])
                compressed_nonzero = torch.count_nonzero(compressed[key])
                
                # Allow some tolerance in the ratio
                expected_nonzero = int(original_nonzero * ratio * 1.2)  # Add 20% tolerance
                assert compressed_nonzero <= expected_nonzero, \
                    f"Too many non-zero values after compression: {compressed_nonzero} vs expected {expected_nonzero}"
                
                print(f"Compression ratio {ratio}: Original non-zeros: {original_nonzero}, "
                      f"Compressed non-zeros: {compressed_nonzero}")
                break
    
    print("Model compression test passed!")

def test_non_iid_distribution():
    """Test non-IID data distribution"""
    # Load MNIST dataset
    train_data, _, _ = load_mnist_data()
    
    num_clients = 5
    
    # Test with different alpha values (concentration parameters)
    for alpha in [0.1, 1.0, 5.0]:
        print(f"\nTesting non-IID distribution with alpha={alpha}")
        
        # Get client indices using Dirichlet distribution
        client_indices = create_non_iid_data_indices(train_data, num_clients, alpha=alpha)
        
        # Check if each client has data
        for i, indices in enumerate(client_indices):
            assert len(indices) > 0, f"Client {i} has no data"
            print(f"Client {i}: {len(indices)} samples")
        
        # For each client, count samples per class to verify skew
        class_distributions = []
        
        for client_idx, indices in enumerate(client_indices):
            # Extract labels for this client's data
            if hasattr(train_data, 'targets'):
                if torch.is_tensor(train_data.targets):
                    client_labels = train_data.targets[indices].numpy()
                else:
                    client_labels = np.array([train_data.targets[i] for i in indices])
            else:
                # Assuming TensorDataset with one-hot encoded labels
                client_labels = torch.argmax(train_data.tensors[1][indices], dim=1).numpy()
            
            # Count labels
            unique_labels, counts = np.unique(client_labels, return_counts=True)
            distribution = np.zeros(10)  # Assuming 10 classes for MNIST
            distribution[unique_labels] = counts
            class_distributions.append(distribution)
            
            print(f"Client {client_idx} class distribution: {distribution}")
        
        # Compute Kullback-Leibler divergence between clients as a measure of non-IID-ness
        from scipy.special import rel_entr
        
        total_divergence = 0
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                # Normalize distributions
                p = class_distributions[i] / np.sum(class_distributions[i])
                q = class_distributions[j] / np.sum(class_distributions[j])
                
                # Add small epsilon to avoid division by zero
                p = p + 1e-10
                q = q + 1e-10
                p = p / np.sum(p)
                q = q / np.sum(q)
                
                # Compute KL divergence
                kl_div = np.sum(rel_entr(p, q))
                total_divergence += kl_div
        
        avg_divergence = total_divergence / (num_clients * (num_clients - 1) / 2)
        print(f"Average KL divergence between clients: {avg_divergence:.4f}")
        
        # Lower alpha should result in higher divergence (more non-IID)
        if alpha == 0.1:
            high_alpha_divergence = avg_divergence
        elif alpha == 5.0:
            low_alpha_divergence = avg_divergence
            # Verify that lower alpha creates more skew
            assert high_alpha_divergence > low_alpha_divergence, \
                "Expected higher divergence with lower alpha"
    
    print("Non-IID distribution test passed!")

if __name__ == "__main__":
    print("Testing new federated learning features...")
    
    test_client_selection()
    test_model_compression()
    test_non_iid_distribution()
    
    print("\nAll tests passed!")

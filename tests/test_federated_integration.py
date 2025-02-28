
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from model import SimpleConvNet
from federated_learning import FederatedLearning
from data_handler import load_mnist_data
from torch.utils.data import TensorDataset, Subset

def create_synthetic_data(num_samples=1000, num_classes=10):
    """Create synthetic data for testing"""
    # Create random features
    features = torch.randn(num_samples, 1, 28, 28)
    
    # Create random one-hot encoded labels
    labels = torch.zeros(num_samples, num_classes)
    random_classes = torch.randint(0, num_classes, (num_samples,))
    for i, c in enumerate(random_classes):
        labels[i, c] = 1.0
    
    return TensorDataset(features, labels)

def test_end_to_end_training():
    """Test end-to-end federated learning training process"""
    print("Running end-to-end training test...")
    
    # Create synthetic data to speed up testing
    train_data = create_synthetic_data(num_samples=500)
    val_data = create_synthetic_data(num_samples=100)
    test_data = create_synthetic_data(num_samples=100)
    
    # Initialize model and federated learning system
    model = SimpleConvNet()
    num_clients = 3
    config = {
        'privacy': {
            'enabled': True,
            'noise_scale': 0.01,
            'privacy_budget': 1.0
        },
        'non_iid': {
            'enabled': False
        },
        'batch_size': 50,
        'device': 'cpu',  # Use CPU for testing
        'verbose': False,  # Disable verbose output for cleaner test logs
        'seed': 42
    }
    
    fl_system = FederatedLearning(model, num_clients=num_clients, config=config)
    
    # Perform one round of training
    accuracy, privacy_loss = fl_system.train_round(
        train_data, 
        val_data, 
        test_data, 
        local_epochs=1
    )
    
    # Check that accuracy is reasonable (non-negative and <= 100)
    assert 0 <= accuracy <= 100, f"Accuracy should be between 0 and 100, got {accuracy}"
    
    # Check that privacy loss is non-negative
    assert privacy_loss >= 0, f"Privacy loss should be non-negative, got {privacy_loss}"
    
    print(f"Training completed with accuracy: {accuracy:.2f}%, privacy loss: {privacy_loss:.4f}")
    print("End-to-end training test passed!")

def test_client_dataset_distribution():
    """Test that data is correctly distributed among clients"""
    print("Testing client dataset distribution...")
    
    # Create synthetic data
    dataset = create_synthetic_data(num_samples=1000)
    
    # Initialize model and federated learning system
    model = SimpleConvNet()
    num_clients = 5
    fl_system = FederatedLearning(model, num_clients=num_clients)
    
    # Test IID distribution
    iid_client_datasets = fl_system.distribute_data(dataset, use_non_iid=False)
    
    # Check that all clients received data
    for i, client_data in enumerate(iid_client_datasets):
        assert len(client_data) > 0, f"Client {i} has no data in IID distribution"
    
    # Check that all data was distributed (sum of client dataset sizes should equal original dataset size)
    total_distributed = sum(len(client_data) for client_data in iid_client_datasets)
    assert total_distributed == len(dataset), f"Expected {len(dataset)} samples, distributed {total_distributed}"
    
    # Test non-IID distribution
    non_iid_client_datasets = fl_system.distribute_data(dataset, use_non_iid=True, alpha=0.5)
    
    # Check that all clients received data
    for i, client_data in enumerate(non_iid_client_datasets):
        assert len(client_data) > 0, f"Client {i} has no data in non-IID distribution"
    
    # Check that all data was distributed
    total_distributed = sum(len(client_data) for client_data in non_iid_client_datasets)
    assert total_distributed == len(dataset), f"Expected {len(dataset)} samples, distributed {total_distributed}"
    
    print("Client dataset distribution test passed!")

def test_model_state_aggregation():
    """Test that model states are correctly aggregated"""
    print("Testing model state aggregation...")
    
    # Initialize model and federated learning system
    model = SimpleConvNet()
    num_clients = 3
    fl_system = FederatedLearning(model, num_clients=num_clients)
    
    # Create synthetic model states
    base_state = model.state_dict()
    client_states = []
    
    # Create three different client states
    for i in range(3):
        client_state = {}
        for key, tensor in base_state.items():
            if isinstance(tensor, torch.Tensor):
                # Create tensors with different known values for each client
                client_state[key] = torch.ones_like(tensor) * (i + 1)
            else:
                client_state[key] = tensor
        client_states.append(client_state)
    
    # Aggregate without privacy or compression
    fl_system.privacy_enabled = False
    aggregated_state = fl_system.aggregate_models(client_states)
    
    # Check that aggregation is correct (should be average of client states)
    for key, tensor in aggregated_state.items():
        if isinstance(tensor, torch.Tensor):
            expected_value = 2.0  # Average of 1, 2, and 3
            assert torch.allclose(tensor, torch.ones_like(tensor) * expected_value), \
                f"Expected aggregated value {expected_value}, got {tensor.mean().item()}"
    
    # Test with compression
    aggregated_compressed = fl_system.aggregate_models(
        client_states, 
        use_compression=True,
        compression_ratio=0.5
    )
    
    # Check that all keys are preserved
    assert set(aggregated_compressed.keys()) == set(base_state.keys()), \
        "Keys don't match after compression-based aggregation"
    
    print("Model state aggregation test passed!")

if __name__ == "__main__":
    print("Running federated learning integration tests...")
    
    test_client_dataset_distribution()
    test_model_state_aggregation()
    test_end_to_end_training()
    
    print("\nAll integration tests passed!")

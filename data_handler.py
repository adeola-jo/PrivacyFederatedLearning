"""
Data handling module for the federated learning framework.
Provides utilities for loading and preprocessing the MNIST dataset
for distributed training across multiple clients.
"""


import torch
import time
from torchvision import datasets
from torch.utils.data import TensorDataset

def dense_to_one_hot(y, class_count):
    """Convert class indices to one-hot encoded tensors using PyTorch"""
    return torch.eye(class_count)[y]

def load_mnist_data(train_val_split_ratio=0.9, num_classes=10, data_dir='./data'):
    """Load and preprocess MNIST dataset with train-validation split
    
    This function uses PyTorch operations throughout for consistency:
    - Normalizes by mean subtraction
    - Converts labels to one-hot encoding
    - Seeds random number generator with current time
    
    Args:
        train_val_split_ratio (float): Ratio of training data to use for training (default: 0.9)
            The remaining data will be used for validation
        num_classes (int): Number of classes for one-hot encoding (default: 10 for MNIST)
        data_dir (str): Directory where the dataset will be stored (default: './data')
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) as TensorDatasets
    """
    # Set random seed based on time
    torch.manual_seed(int(time.time() * 1e6) % 2**31)
    
    # Load datasets without transformation
    train_dataset = datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True)
    
    # Convert to tensors and reshape to [N, 1, 28, 28]
    train_x = train_dataset.data.reshape(-1, 1, 28, 28).float() / 255
    train_y = train_dataset.targets
    
    # Calculate split point based on ratio
    train_size = int(len(train_x) * train_val_split_ratio)
    
    # Split into train and validation sets
    train_x, valid_x = train_x[:train_size], train_x[train_size:]
    train_y, valid_y = train_y[:train_size], train_y[train_size:]
    
    # Process test set
    test_x = test_dataset.data.reshape(-1, 1, 28, 28).float() / 255
    test_y = test_dataset.targets
    
    # Calculate and subtract train mean (using PyTorch operations)
    train_mean = train_x.mean()
    train_x = train_x - train_mean
    valid_x = valid_x - train_mean
    test_x = test_x - train_mean
    
    # Convert labels to one-hot encoding using the specified number of classes
    train_y = dense_to_one_hot(train_y, num_classes)
    valid_y = dense_to_one_hot(valid_y, num_classes)
    test_y = dense_to_one_hot(test_y, num_classes)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(valid_x, valid_y)
    test_dataset = TensorDataset(test_x, test_y)
    
    return train_dataset, val_dataset, test_dataset



# ------------------- OLD IMPLEMENTATION -------------------

# import torch
# from torchvision import datasets, transforms

# def load_mnist_data():
#     """Load and preprocess MNIST dataset"""
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
    
#     train_dataset = datasets.MNIST(
#         './data', 
#         train=True, 
#         download=True,
#         transform=transform
#     )
    
#     test_dataset = datasets.MNIST(
#         './data', 
#         train=False,
#         download=True,
#         transform=transform
#     )
    
#     return train_dataset, test_dataset

"""
Data handling utilities for loading and preprocessing datasets.
"""

import torch
import numpy as np
import torchvision

def load_mnist_data(data_dir='./data', batch_size=64, iid=True, alpha=0.5):
    """
    Load and preprocess the MNIST dataset.

    Args:
        data_dir: Directory to store dataset
        batch_size: Batch size for dataloaders
        iid: If True, data is split IID; if False, non-IID split
        alpha: Dirichlet alpha parameter for non-IID split

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load train and test datasets
    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )

    # Split training set into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    return train_dataset, val_dataset, test_dataset

def dense_to_one_hot(y, class_count):
    """Convert class indices to one-hot encoded tensors using PyTorch"""
    return torch.eye(class_count)[y]

def create_non_iid_data_indices(dataset, num_clients, num_classes=10, alpha=0.5):
    """
    Create non-IID data distribution using a Dirichlet distribution

    Args:
        dataset: Dataset to partition
        num_clients (int): Number of clients
        num_classes (int): Number of classes
        alpha (float): Dirichlet concentration parameter. Lower alpha means more skewed distribution.

    Returns:
        list: List of indices for each client
    """
    import numpy as np

    # Get dataset targets/labels
    if hasattr(dataset, 'targets'):
        labels = dataset.targets.numpy() if torch.is_tensor(dataset.targets) else np.array(dataset.targets)
    elif hasattr(dataset, 'tensors'):
        # For TensorDataset with one-hot encoded labels, get the class index
        labels = torch.argmax(dataset.tensors[1], dim=1).numpy()
    else:
        raise ValueError("Dataset format not recognized")

    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]

    # Group indices by label
    label_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}

    # For each class, distribute indices to clients according to Dirichlet distribution
    for class_idx in range(num_classes):
        # Get indices for this class
        idx_for_class = label_indices[class_idx]

        # Skip if no samples for this class
        if len(idx_for_class) == 0:
            continue

        # Generate Dirichlet distribution for this class
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Calculate number of samples per client for this class
        num_samples_per_client = np.array([int(p * len(idx_for_class)) for p in proportions])

        # Adjust to make sure all samples are assigned
        diff = len(idx_for_class) - np.sum(num_samples_per_client)
        num_samples_per_client[0] += diff

        # Assign indices to clients
        start_idx = 0
        for client_idx in range(num_clients):
            end_idx = start_idx + num_samples_per_client[client_idx]
            if start_idx < end_idx:  # Only add if there are samples to add
                client_indices[client_idx].extend(idx_for_class[start_idx:end_idx])
            start_idx = end_idx

    return client_indices

def distribute_data_non_iid(dataset, num_clients, alpha=0.5):
    """
    Distribute data in a non-IID manner among clients using Dirichlet distribution

    Args:
        dataset: Dataset to distribute
        num_clients (int): Number of clients
        alpha (float): Dirichlet concentration parameter

    Returns:
        list: List of Subset datasets, one for each client
    """
    from torch.utils.data import Subset

    client_indices = create_non_iid_data_indices(dataset, num_clients, alpha=alpha)
    return [Subset(dataset, indices) for indices in client_indices]

# src/utils/__init__.py
#Empty file to make src.utils a package

# src/utils/database.py (Assumed implementation)
def get_db():
    """Placeholder for database connection"""
    return "Database Connection"

# src/__init__.py
#Empty file to make src a package

# data_handler.py (added based on context)
import torch
import numpy as np
import torchvision
from ..utils.database import get_db


def some_function():
    db_connection = get_db()
    print(f"Database Connection: {db_connection}")
import torch
from torchvision import datasets, transforms

def load_mnist_data():
    """Load and preprocess MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        './data', 
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset

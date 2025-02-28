#!/usr/bin/env python3
"""
Basic example of using the Privacy-Preserving Federated Learning framework.
This example shows how to set up and run a federated learning experiment
programmatically, without using the Streamlit UI.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models.model import SimpleConvNet
from core.federated_learning import FederatedLearning
from data.data_handler import load_mnist_data
import torch

def main():
    # Load data
    print("Loading MNIST dataset...")
    train_data, val_data, test_data = load_mnist_data(iid=True)
    
    # Initialize model
    model = SimpleConvNet()
    
    # Configure federated learning
    config = {
        'privacy': {
            'enabled': True,
            'noise_scale': 0.1,
            'privacy_budget': 1.0
        },
        'compression': {
            'enabled': False,
            'ratio': 0.5
        },
        'non_iid': {
            'enabled': False,
            'alpha': 0.5
        },
        'batch_size': 64,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'verbose': True
    }
    
    # Initialize federated learning system
    fl_system = FederatedLearning(
        model, 
        num_clients=5,
        config=config
    )
    
    # Run federated learning for 5 rounds
    print("Starting federated learning...")
    for round_idx in range(5):
        # Perform one round of federated learning
        round_accuracy, privacy_loss = fl_system.train_round(
            train_data,
            val_data,
            test_data,
            local_epochs=2,
            client_fraction=0.6
        )
        
        print(f"Round {round_idx+1}/5 - Accuracy: {round_accuracy:.2f}%, Privacy Loss: {privacy_loss:.4f}")
    
    # Evaluate final model
    final_accuracy = fl_system.evaluate(test_data)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()


# Privacy-Preserving Federated Learning Framework

A comprehensive framework for implementing privacy-preserving federated learning using PyTorch and Streamlit. This project demonstrates secure distributed machine learning while maintaining data privacy through differential privacy techniques.

## Features

- Federated Learning with multiple clients
- Differential Privacy implementation
- Real-time training visualization
- Experiment tracking and storage
- Interactive parameter tuning
- MNIST dataset demonstration

## Structure

- `main.py`: Streamlit web interface and experiment orchestration
- `federated_learning.py`: Core federated learning implementation
- `model.py`: Neural network model architecture
- `differential_privacy.py`: Privacy preservation utilities
- `data_handler.py`: Dataset loading and preprocessing
- `visualization.py`: Training progress visualization
- `database.py`: Experiment tracking and storage

## Usage

1. Configure experiment parameters in the sidebar
2. Start training with the "Start Training" button
3. Monitor training progress and privacy metrics
4. View previous experiment results

## Technical Details

The framework implements federated averaging (FedAvg) with differential privacy guarantees. Each client trains locally, and models are aggregated with noise addition for privacy preservation.

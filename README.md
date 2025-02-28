
# Privacy-Preserving Federated Learning Framework

A comprehensive framework for implementing privacy-preserving federated learning using PyTorch and Streamlit. This project demonstrates secure distributed machine learning while maintaining data privacy through differential privacy techniques.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Implementation](#technical-implementation)
- [Contributing](#contributing)

## Overview

Federated Learning is a machine learning approach that trains an algorithm across multiple decentralized devices or servers holding local data samples, without exchanging them. This approach enables multiple participants to build a common, robust machine learning model without sharing data, thus addressing critical issues such as data privacy, data security, data access rights, and access to heterogeneous data.

This framework implements a privacy-preserving federated learning system that incorporates differential privacy to provide formal privacy guarantees while maintaining high model utility.

## Features

- **Federated Learning**: Implementation of FedAvg algorithm across multiple clients
- **Differential Privacy**: Noise addition mechanisms to provide formal privacy guarantees
- **Privacy Budget Monitoring**: Real-time tracking of privacy loss throughout training
- **Non-IID Data Support**: Realistic data distribution simulation using Dirichlet distributions
- **Model Compression**: Weight pruning to reduce communication overhead
- **Interactive Web Interface**: Streamlit-based dashboard for experiment configuration and visualization
- **Experiment Tracking**: Database storage and comparison of experiment results
- **Real-time Visualization**: Interactive charts for monitoring training progress
- **Model Testing**: Visual inspection of model predictions on test data

## Installation

The project runs on Python 3.8+ and requires the following main dependencies:
- PyTorch
- Streamlit
- SQLAlchemy
- Plotly
- NumPy
- Matplotlib

### Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd privacy-preserving-federated-learning
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run main.py
   ```

## Usage

The application provides an intuitive web interface that can be accessed at `http://localhost:5000` after starting the application.

### Configuration

1. Use the **Configuration** tab to set parameters:
   - Number of clients participating in federated learning
   - Client participation fraction
   - Number of federated rounds
   - Local training epochs
   - Privacy budget and noise scale for differential privacy
   - Data distribution settings (IID or non-IID)
   - Model compression ratio

2. Optionally add a description for your experiment

### Training

1. Navigate to the **Training** tab
2. Click the "Start Training" button to begin federated learning
3. Monitor training progress and privacy metrics in real-time
4. After training completes, test the global model on the test dataset
5. View sample predictions on test images

### Experiment Comparison

1. Use the **Experiment Comparison** tab to compare results from different experiments
2. Select multiple experiments to visualize their performance metrics together
3. Analyze the impact of different parameter settings on model accuracy and privacy loss

## Project Structure

```
privacy-preserving-federated-learning/
│
├── main.py                 # Main application and UI (Streamlit)
├── model.py                # Neural network model architecture
├── federated_learning.py   # Core federated learning implementation
├── differential_privacy.py # Privacy mechanisms
├── data_handler.py         # Dataset loading and processing
├── database.py             # Database models and utilities
├── visualization.py        # Plotting and visualization functions
├── federated_utils.py      # Utility functions for federated learning
│
├── tests/                  # Unit and integration tests
│   ├── test_federated_integration.py
│   ├── test_differential_privacy.py
│   ├── test_db_operations.py
│   └── test_new_features.py
│
└── data/                   # Data storage directory
    └── MNIST/              # MNIST dataset files
```

## Technical Implementation

### Federated Learning

The framework implements the Federated Averaging (FedAvg) algorithm, which consists of:
1. Distributing the global model to selected clients
2. Training local models on client data
3. Aggregating local model updates to improve the global model
4. Repeating the process for multiple rounds

```python
# Pseudo-code for FedAvg
for each round t = 1, 2, ...:
    select a subset of clients
    for each selected client i:
        client_update = client_training(global_model, local_data)
    global_model = aggregate(client_updates)
```

### Differential Privacy

To provide privacy guarantees, the framework adds calibrated Gaussian noise to model updates:

```python
def add_noise(tensor, noise_scale):
    """Add Gaussian noise to tensor for differential privacy."""
    if noise_scale <= 0:
        return tensor
    return tensor + torch.normal(0, noise_scale, tensor.shape)
```

The framework tracks privacy loss using a simplified privacy accounting method.

### Non-IID Data Distribution

The framework uses a Dirichlet distribution to create realistic non-IID data partitions:

```python
def distribute_data_non_iid(dataset, num_clients, alpha=0.5):
    """Distribute data among clients according to a Dirichlet distribution."""
    labels = get_dataset_labels(dataset)
    label_indices = group_by_label(labels)
    client_data_indices = partition_by_dirichlet(label_indices, num_clients, alpha)
    return [Subset(dataset, indices) for indices in client_data_indices]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

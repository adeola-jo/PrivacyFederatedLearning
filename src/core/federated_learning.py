"""
Core implementation of federated learning algorithms.
Handles client creation, model aggregation, and differential privacy.
"""

import torch
import numpy as np
import copy
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from privacy.differential_privacy import add_noise, calculate_privacy_loss

class FederatedLearning:
    """
    Implements federated learning with privacy-preserving mechanisms.
    """

    def __init__(self, model, num_clients, config=None):
        """
        Initialize the federated learning system.

        Args:
            model: PyTorch model to be trained
            num_clients: Number of clients/participants
            config: Dictionary with configuration parameters
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default configuration
        default_config = {
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
            'device': self.device,
            'verbose': True
        }

        # Update with user config
        self.config = default_config
        if config:
            for category in config:
                if isinstance(config[category], dict) and category in self.config:
                    # Merge dict configs
                    self.config[category].update(config[category])
                else:
                    # Set non-dict configs directly
                    self.config[category] = config[category]

        # Initialize model and clients
        self.global_model = model.to(self.device)
        self.num_clients = num_clients
        self.clients = self._create_clients()

        # Initialize privacy tracking
        self.total_privacy_loss = 0.0
        self.privacy_losses_per_round = []

    def _create_clients(self):
        """Create client models by copying the global model."""
        return [copy.deepcopy(self.global_model) for _ in range(self.num_clients)]

    def _split_data_for_clients(self, train_data, iid=True, alpha=0.5):
        """
        Split the training data among clients.

        Args:
            train_data: The training dataset
            iid: If True, data is split IID; if False, non-IID split
            alpha: Dirichlet alpha parameter for non-IID split

        Returns:
            List of datasets, one for each client
        """
        num_samples = len(train_data)
        indices = list(range(num_samples))

        # IID split: simple random division
        if iid:
            np.random.shuffle(indices)
            client_indices = np.array_split(indices, self.num_clients)
        # Non-IID split: Dirichlet distribution
        else:
            labels = np.array([train_data[i][1] for i in range(num_samples)])
            unique_labels = np.unique(labels)
            client_indices = [[] for _ in range(self.num_clients)]

            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                np.random.shuffle(label_indices)

                # Dirichlet distribution for label assignment
                proportions = np.random.dirichlet(np.repeat(alpha, self.num_clients))
                proportions = np.array([p*(len(label_indices)//self.num_clients) for p in proportions])
                proportions = np.round(proportions).astype(int)
                proportions[0] += len(label_indices) - np.sum(proportions)

                # Distribute indices according to proportions
                start_idx = 0
                for client_idx, prop in enumerate(proportions):
                    client_indices[client_idx].extend(label_indices[start_idx:start_idx+prop].tolist())
                    start_idx += prop

        # Create subdatasets for clients
        client_datasets = []
        for indices in client_indices:
            client_datasets.append(torch.utils.data.Subset(train_data, indices))

        return client_datasets

    def _train_client(self, client_model, client_data, epochs=1):
        """
        Train a client model on its local data.

        Args:
            client_model: The client's model
            client_data: The client's dataset
            epochs: Number of training epochs

        Returns:
            The updated client model and number of samples trained on
        """
        # Set up training
        client_model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            client_model.parameters(), 
            lr=0.01, 
            momentum=0.9
        )

        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            client_data, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )

        # Train for specified epochs
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = client_model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

        return client_model, len(client_data)

    def _aggregate_models(self, client_models, client_sizes):
        """
        Aggregate client models into the global model.

        Args:
            client_models: List of client models
            client_sizes: Number of samples per client

        Returns:
            The aggregated global model
        """
        # Create a copy of the current global model
        global_model = copy.deepcopy(self.global_model)

        # Get total number of samples across all clients
        total_size = sum(client_sizes)

        # Initialize global parameters with zeros
        for param in global_model.parameters():
            param.data.zero_()

        # Weighted aggregation of client models
        for i, (client_model, size) in enumerate(zip(client_models, client_sizes)):
            # Calculate weight for this client (proportion of data)
            weight = size / total_size

            # Apply differential privacy if enabled
            if self.config['privacy']['enabled']:
                noise_scale = self.config['privacy']['noise_scale']
                if noise_scale > 0:
                    # Add noise to client model parameters
                    for param in client_model.parameters():
                        param.data = add_noise(param.data, noise_scale)

            # Apply model compression if enabled
            if self.config['compression']['enabled']:
                compression_ratio = self.config['compression']['ratio']
                if compression_ratio < 1.0:
                    for param in client_model.parameters():
                        # Keep only top k% of parameters by magnitude
                        flat_param = param.data.flatten()
                        k = int(flat_param.numel() * compression_ratio)
                        values, indices = torch.topk(torch.abs(flat_param), k)
                        mask = torch.zeros_like(flat_param)
                        mask[indices] = 1
                        param.data = (param.data * mask.view(param.data.shape))

            # Add weighted client parameters to global parameters
            for global_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                global_param.data += weight * client_param.data

        return global_model

    def train_round(self, train_data, val_data, test_data, local_epochs=1, client_fraction=1.0):
        """
        Perform one round of federated learning.

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset
            local_epochs: Number of epochs for local client training
            client_fraction: Fraction of clients to include in this round

        Returns:
            Tuple of (round_accuracy, privacy_loss)
        """
        # Split data among clients
        client_datasets = self._split_data_for_clients(
            train_data, 
            iid=not self.config['non_iid']['enabled'],
            alpha=self.config['non_iid']['alpha']
        )

        # Select random subset of clients according to fraction
        num_selected = max(1, int(self.num_clients * client_fraction))
        selected_indices = np.random.choice(self.num_clients, num_selected, replace=False)

        # Train selected clients
        selected_models = []
        selected_sizes = []

        for idx in selected_indices:
            # Create a fresh copy of the global model for this client
            client_model = copy.deepcopy(self.global_model)

            # Train the client and get updated model
            updated_model, num_samples = self._train_client(
                client_model, 
                client_datasets[idx],
                local_epochs
            )

            selected_models.append(updated_model)
            selected_sizes.append(num_samples)

        # Calculate privacy loss for this round if enabled
        if self.config['privacy']['enabled']:
            round_privacy_loss = calculate_privacy_loss(
                self.config['privacy']['noise_scale'],
                num_selected,
                self.num_clients
            )
            self.total_privacy_loss += round_privacy_loss
            self.privacy_losses_per_round.append(round_privacy_loss)
        else:
            round_privacy_loss = 0.0
            self.privacy_losses_per_round.append(0.0)

        # Check if privacy budget is exceeded
        if (self.config['privacy']['enabled'] and 
            self.total_privacy_loss > self.config['privacy']['privacy_budget']):
            print(f"Warning: Privacy budget ({self.config['privacy']['privacy_budget']}) exceeded: {self.total_privacy_loss}")

        # Aggregate models
        self.global_model = self._aggregate_models(selected_models, selected_sizes)

        # Evaluate
        accuracy = self.evaluate(test_data)

        return accuracy, round_privacy_loss

    def evaluate(self, test_data):
        """
        Evaluate the global model on test data.

        Args:
            test_data: Test dataset

        Returns:
            Accuracy percentage
        """
        # Set up evaluation
        self.global_model.eval()
        correct = 0
        total = 0
        test_loader = torch.utils.data.DataLoader(
            test_data, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )

        # Evaluate model
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.global_model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy
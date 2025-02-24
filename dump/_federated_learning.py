"""
Core implementation of the Federated Learning system.
Handles client model distribution, training, and aggregation with privacy preservation.
Implements the FedAvg algorithm with differential privacy guarantees.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from differential_privacy import add_noise

class FederatedLearning:
    """
    Federated Learning class to manage distributed model training with differential privacy.

    Args:
        model (torch.nn.Module): The global model to be trained.
        num_clients (int): The number of clients participating in federated learning.
        privacy_budget (float): The total privacy budget for the training process.
        noise_scale (float): The scale of noise added for differential privacy.
    """
    def __init__(self, model, num_clients, privacy_budget, noise_scale):
        self.model = model
        self.num_clients = num_clients
        self.privacy_budget = privacy_budget
        self.noise_scale = noise_scale
        self.client_models = [type(model)() for _ in range(num_clients)]

        #TODO: Collect all criterions parameters togetther
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

    def distribute_data(self, dataset):
        """Split dataset among clients"""
        total_size = len(dataset)
        indices = np.random.permutation(total_size)
        client_data = np.array_split(indices, self.num_clients)
        return [Subset(dataset, indices) for indices in client_data]

    
    #TODO: Edit the client training code
    def train_client(self, client_id, client_data, local_epochs):
        """Train a client model"""
        model = self.client_models[client_id]
        model.load_state_dict(self.model.state_dict())
        optimizer = torch.optim.Adam(model.parameters())

        dataloader = DataLoader(client_data, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(local_epochs):
            for data, target in dataloader:
                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

        return model.state_dict()
    


    def aggregate_models(self, client_states):
        """Aggregate client models using FedAvg"""
        aggregated_state = {}
        for key in client_states[0].keys():
            aggregated_state[key] = torch.stack([
                add_noise(states[key], self.noise_scale)
                for states in client_states
            ]).mean(0)
        return aggregated_state

    def train_round(self, train_data, test_data, local_epochs):
        """Perform one round of federated learning
            by distributing the training data across
            clients, training each client, aggregating
            the models, and evaluating the global model.

        Args:
            train_data (torch.utils.data.Dataset): The training dataset
            test_data (torch.utils.data.Dataset): The test dataset
            local_epochs (int): Number of local training epochs

        Returns:
            tuple: (accuracy, privacy_loss) after the round

        """
        client_datasets = self.distribute_data(train_data)

        # Train clients
        client_states = []
        for client_id in range(self.num_clients):
            client_state = self.train_client(
                client_id, 
                client_datasets[client_id],
                local_epochs
            )
            client_states.append(client_state)

        # Aggregate models
        aggregated_state = self.aggregate_models(client_states)
        self.model.load_state_dict(aggregated_state)

        # Evaluate
        accuracy = self.evaluate(test_data)
        privacy_loss = self.calculate_privacy_loss()

        return accuracy, privacy_loss

    def evaluate(self, test_data):
        """Evaluate model on test data"""
        self.model.eval()
        correct = 0
        total = 0

        dataloader = DataLoader(test_data, batch_size=32)

        with torch.no_grad():
            for data, target in dataloader:
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)

        return 100. * correct / total

    def calculate_privacy_loss(self):
        """Calculate privacy loss based on noise scale and budget"""
        return self.privacy_budget * (1 - np.exp(-1/self.noise_scale))
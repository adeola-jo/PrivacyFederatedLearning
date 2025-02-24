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

        # Hyperparameters for training
        self.config = {
            'max_epochs': 8,
            'batch_size': 50,
            'save_dir': 'weights',
            'weight_decay': 1e-2,
            'optimizer': 'sgd',  # Options: 'sgd', 'adam', 'rmsprop'
            'lr_scheduler': 'policy',  # Options: 'policy', 'step', 'exponential', 'cosine', 'none'
            'lr_scheduler_params': {
                'step': {'step_size': 2, 'gamma': 0.1},
                'exponential': {'gamma': 0.9},
                'cosine': {'T_max': 8, 'eta_min': 1e-4}
            },
            'lr_policy': {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}}
        }

        # Initialize criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize the global optimizer based on config
        self.optimizer = self._create_optimizer(
            self.model.parameters(), 
            self.config['optimizer'],
            lr=self.config['lr_policy'][1]['lr'],
            weight_decay=self.config['weight_decay']
        )

    def _create_optimizer(self, parameters, optimizer_type, **kwargs):
        """
        Create an optimizer based on the specified type
        
        Args:
            parameters: Model parameters to optimize
            optimizer_type (str): Type of optimizer ('sgd', 'adam', 'rmsprop')
            **kwargs: Additional optimizer parameters
            
        Returns:
            torch.optim.Optimizer: The created optimizer
        """
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'sgd':
            return torch.optim.SGD(parameters, **kwargs)
        elif optimizer_type == 'adam':
            return torch.optim.Adam(parameters, **kwargs)
        elif optimizer_type == 'rmsprop':
            return torch.optim.RMSprop(parameters, **kwargs)
        else:
            print(f"Warning: Unknown optimizer type '{optimizer_type}'. Using SGD as default.")
            return torch.optim.SGD(parameters, **kwargs)
            
    def _create_scheduler(self, optimizer, scheduler_type, params=None):
        """
        Create a learning rate scheduler based on the specified type
        
        Args:
            optimizer: The optimizer to schedule
            scheduler_type (str): Type of scheduler ('policy', 'step', 'exponential', 'cosine', 'none')
            params (dict): Parameters for the scheduler
            
        Returns:
            torch.optim.lr_scheduler._LRScheduler or None: The created scheduler
        """
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == 'none' or scheduler_type is None:
            return None
        elif scheduler_type == 'policy':
            # This is handled manually in the train_client method
            return None
        elif scheduler_type == 'step':
            step_size = params.get('step_size', 2)
            gamma = params.get('gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'exponential':
            gamma = params.get('gamma', 0.9)
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_type == 'cosine':
            T_max = params.get('T_max', 8)
            eta_min = params.get('eta_min', 0)
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        else:
            print(f"Warning: Unknown scheduler type '{scheduler_type}'. No scheduler will be used.")
            return None

    def distribute_data(self, dataset):
        """Split dataset among clients"""
        total_size = len(dataset)
        indices = np.random.permutation(total_size)
        client_data = np.array_split(indices, self.num_clients)
        return [Subset(dataset, indices) for indices in client_data]

    def train_client(self, client_id, client_data, val_data, local_epochs):
        """
        Train a client model with proper learning rate policy and validation
        
        Args:
            client_id (int): The ID of the client being trained
            client_data (Dataset): The client's training data
            val_data (Dataset): Validation dataset
            local_epochs (int): Number of local training epochs
            
        Returns:
            dict: The trained model's state dictionary
        """
        # Set up the client model
        model = self.client_models[client_id]
        model.load_state_dict(self.model.state_dict())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Setup data loaders
        train_loader = DataLoader(
            client_data, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.config['batch_size']
        )
        
        # Get initial learning rate
        initial_lr = self.config['lr_policy'][1]['lr']
        
        # Create optimizer
        optimizer = self._create_optimizer(
            model.parameters(),
            self.config['optimizer'],
            lr=initial_lr,
            weight_decay=self.config['weight_decay']
        )
        
        # Create scheduler if requested
        scheduler_type = self.config['lr_scheduler']
        scheduler = None
        
        if scheduler_type != 'policy' and scheduler_type != 'none':
            scheduler_params = self.config['lr_scheduler_params'].get(scheduler_type, {})
            scheduler = self._create_scheduler(optimizer, scheduler_type, scheduler_params)
        
        # Train the model
        model.train()
        
        for epoch in range(1, local_epochs + 1):
            # For manual policy-based learning rate adjustment
            if scheduler_type == 'policy':
                # Get learning rate for this epoch from policy
                epoch_key = epoch
                if epoch_key not in self.config['lr_policy']:
                    # Find the closest lr policy epoch that's smaller
                    available_epochs = sorted([k for k in self.config['lr_policy'].keys() if k <= epoch_key])
                    if available_epochs:
                        epoch_key = available_epochs[-1]
                    else:
                        epoch_key = min(self.config['lr_policy'].keys())
                
                lr = self.config['lr_policy'][epoch_key]['lr']
                
                # Update optimizer with current learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Training loop
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Print progress every 10 batches
                if batch_idx % 10 == 0:
                    print(f"Client {client_id}, Epoch {epoch}, Batch {batch_idx}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Acc: {100. * correct / total:.2f}%")
            
            # Step the scheduler after each epoch if using a PyTorch scheduler
            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Client {client_id}, Epoch {epoch}, LR: {current_lr:.6f}")
            
            # Validation after each epoch
            val_loss, val_acc = self.evaluate_client(model, val_loader, device)
            print(f"Client {client_id}, Epoch {epoch}, "
                  f"Train Loss: {train_loss / len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%")
        
        return model.state_dict()
    
    def evaluate_client(self, model, data_loader, device):
        """Evaluate a client model on the validation data"""
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = self.criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss = val_loss / len(data_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc

    def aggregate_models(self, client_states):
        """Aggregate client models using FedAvg with differential privacy"""
        aggregated_state = {}
        for key in client_states[0].keys():
            # Add noise to each parameter for differential privacy
            noisy_params = torch.stack([
                add_noise(states[key], self.noise_scale)
                for states in client_states
            ])
            # Average the parameters across clients
            aggregated_state[key] = noisy_params.mean(0)
        
        return aggregated_state

    def train_round(self, train_data, val_data, test_data, local_epochs):
        """Perform one round of federated learning
            by distributing the training data across
            clients, training each client, validating on
            the same dataset, and aggregating
            the models, and evaluating the global model.

        Args:
            train_data (torch.utils.data.Dataset): The training dataset
            val_data (torch.utils.data.Dataset): The validation dataset
            test_data (torch.utils.data.Dataset): The test dataset
            local_epochs (int): Number of local training epochs

        Returns:
            tuple: (accuracy, privacy_loss) after the round
        """
        # Distribute data among clients
        client_datasets = self.distribute_data(train_data)
        
        print(f"Starting federated learning round with {self.num_clients} clients")
        print(f"Local epochs per client: {local_epochs}")

        # Train clients in parallel (simulated sequential here)
        client_states = []
        for client_id in range(self.num_clients):
            print(f"Training client {client_id+1}/{self.num_clients}")
            client_state = self.train_client(
                client_id, 
                client_datasets[client_id],
                val_data,
                local_epochs
            )
            client_states.append(client_state)

        # Aggregate models with differential privacy
        print("Aggregating client models with differential privacy")
        aggregated_state = self.aggregate_models(client_states)
        self.model.load_state_dict(aggregated_state)

        # Evaluate global model on test data
        print("Evaluating global model on test data")
        accuracy = self.evaluate(test_data)
        privacy_loss = self.calculate_privacy_loss()
        
        print(f"Global model accuracy: {accuracy:.2f}%")
        print(f"Privacy loss: {privacy_loss:.4f}")

        return accuracy, privacy_loss

    def evaluate(self, test_data):
        """Evaluate global model on test data"""
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        
        correct = 0
        total = 0
        test_loss = 0

        dataloader = DataLoader(test_data, batch_size=self.config['batch_size'])

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss = test_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%")
        
        return accuracy

    def calculate_privacy_loss(self):
        """Calculate privacy loss based on noise scale and budget"""
        # This is a simple model; real-world implementations would use 
        # more sophisticated privacy accounting methods
        return self.privacy_budget * (1 - np.exp(-1/self.noise_scale))
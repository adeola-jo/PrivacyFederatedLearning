"""
Core implementation of the Federated Learning system.
Handles client model distribution, training, and aggregation with privacy preservation.
Implements the FedAvg algorithm with differential privacy guarantees.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

# Import utility functions and factories
from federated_utils import (
    OptimizerFactory,
    SchedulerFactory,
    LossFactory,
    apply_lr_policy,
    DefaultFederatedConfig
)


class FederatedLearning:
    """
    Federated Learning class to manage distributed model training with differential privacy.
    
    Args:
        model (torch.nn.Module): The global model to be trained.
        num_clients (int): The number of clients participating in federated learning.
        config (dict, optional): Configuration dictionary. If None, default config is used.
    """
    def __init__(self, model, num_clients, config=None):
        self.model = model
        self.num_clients = num_clients
        
        # Load config (use default if none provided)
        self.config = DefaultFederatedConfig.get_config() if config is None else config
        
        # Set random seed for reproducibility
        if 'seed' in self.config:
            torch.manual_seed(self.config['seed'])
            np.random.seed(self.config['seed'])
            
        # Create client models
        self.client_models = [type(model)() for _ in range(num_clients)]
        
        # Set up device
        self.device = torch.device(self.config['device'])
        self.model = self.model.to(self.device)
        
        # Extract privacy settings
        privacy_config = self.config.get('privacy', {})
        self.privacy_enabled = privacy_config.get('enabled', True)
        self.noise_scale = privacy_config.get('noise_scale', 0.01)
        self.privacy_budget = privacy_config.get('privacy_budget', 1.0)
        
        # Initialize criterion
        loss_type = self.config.get('loss', 'cross_entropy')
        loss_params = self.config.get('loss_params', {})
        self.criterion = LossFactory.create(loss_type, **loss_params)
        
        # Initialize the global optimizer
        lr = self.config['lr_policy'][min(self.config['lr_policy'].keys())]['lr']
        optimizer_type = self.config.get('optimizer', 'sgd')
        optimizer_params = self.config.get('optimizer_params', {})
        self.optimizer = OptimizerFactory.create(
            self.model.parameters(),
            optimizer_type,
            lr=lr,
            weight_decay=self.config.get('weight_decay', 1e-2),
            **optimizer_params
        )

    def distribute_data(self, dataset):
        """
        Split dataset among clients
        
        Args:
            dataset (torch.utils.data.Dataset): The dataset to distribute
            
        Returns:
            list: List of datasets, one for each client
        """
        total_size = len(dataset)
        indices = np.random.permutation(total_size)
        client_data = np.array_split(indices, self.num_clients)
        return [Subset(dataset, indices) for indices in client_data]

    def train_client(self, client_id, client_data, val_data, local_epochs):
        """
        Train a client model
        
        Args:
            client_id (int): The ID of the client being trained
            client_data (Dataset): The client's training data
            val_data (Dataset): Validation dataset
            local_epochs (int): Number of local training epochs
            
        Returns:
            dict: The trained model's state dictionary
        """
        verbose = self.config.get('verbose', True)
        
        # Set up the client model
        model = self.client_models[client_id]
        model.load_state_dict(self.model.state_dict())
        model = model.to(self.device)
        
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
        lr_policy = self.config['lr_policy']
        initial_lr = lr_policy[min(lr_policy.keys())]['lr']
        
        # Create optimizer
        optimizer_type = self.config.get('optimizer', 'sgd')
        optimizer_params = self.config.get('optimizer_params', {})
        optimizer = OptimizerFactory.create(
            model.parameters(),
            optimizer_type,
            lr=initial_lr,
            weight_decay=self.config.get('weight_decay', 1e-2),
            **optimizer_params
        )
        
        # Create scheduler if requested
        scheduler_type = self.config.get('lr_scheduler', 'policy')
        scheduler = None
        
        if scheduler_type != 'policy' and scheduler_type != 'none':
            scheduler_params = self.config.get('lr_scheduler_params', {}).get(scheduler_type, {})
            scheduler = SchedulerFactory.create(optimizer, scheduler_type, scheduler_params)
        
        # Train the model
        model.train()
        
        for epoch in range(1, local_epochs + 1):
            # For manual policy-based learning rate adjustment
            if scheduler_type == 'policy':
                current_lr = apply_lr_policy(optimizer, epoch, lr_policy)
                if verbose:
                    print(f"Client {client_id}, Epoch {epoch}, LR: {current_lr:.6f}")
            
            # Training loop
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Print progress periodically
                if verbose and batch_idx % 10 == 0:
                    print(f"Client {client_id}, Epoch {epoch}, Batch {batch_idx}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Acc: {100. * correct / total:.2f}%")
            
            # Step the scheduler after each epoch if using a PyTorch scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # For ReduceLROnPlateau, we need validation loss
                    val_loss, _ = self.evaluate_client(model, val_loader)
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                if verbose:
                    print(f"Client {client_id}, Epoch {epoch}, LR: {current_lr:.6f}")
            
            # Validation after each epoch
            val_loss, val_acc = self.evaluate_client(model, val_loader)
            
            if verbose:
                train_acc = 100. * correct / total
                print(f"Client {client_id}, Epoch {epoch}, "
                      f"Train Loss: {train_loss / len(train_loader):.4f}, "
                      f"Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.2f}%")
        
        return model.state_dict()
    
    def evaluate_client(self, model, data_loader):
        """
        Evaluate a client model on the validation data
        
        Args:
            model (torch.nn.Module): The model to evaluate
            data_loader (DataLoader): The data loader for evaluation
            
        Returns:
            tuple: (loss, accuracy)
        """
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
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
        """
        Aggregate client models using FedAvg with differential privacy
        
        Args:
            client_states (list): List of client model state dictionaries
            
        Returns:
            dict: Aggregated model state dictionary
        """
        aggregated_state = {}
        for key in client_states[0].keys():
            # Stack parameters from all clients
            stacked_params = torch.stack([states[key] for states in client_states])
            
            # Add noise for differential privacy if enabled
            if self.privacy_enabled and self.noise_scale > 0:
                noisy_params = torch.stack([
                    self.add_differential_privacy_noise(states[key], self.noise_scale)
                    for states in client_states
                ])
                # Average the parameters across clients
                aggregated_state[key] = noisy_params.mean(0)
            else:
                # Average the parameters across clients without noise
                aggregated_state[key] = stacked_params.mean(0)
        
        return aggregated_state

    def train_round(self, train_data, val_data, test_data, local_epochs):
        """
        Perform one round of federated learning
        
        Args:
            train_data (Dataset): The training dataset
            val_data (Dataset): The validation dataset
            test_data (Dataset): The test dataset
            local_epochs (int): Number of local training epochs
            
        Returns:
            tuple: (accuracy, privacy_loss) after the round
        """
        verbose = self.config.get('verbose', True)
        
        # Distribute data among clients
        client_datasets = self.distribute_data(train_data)
        
        if verbose:
            print(f"Starting federated learning round with {self.num_clients} clients")
            print(f"Local epochs per client: {local_epochs}")

        # Train clients in parallel (simulated sequential here)
        client_states = []
        for client_id in range(self.num_clients):
            if verbose:
                print(f"Training client {client_id+1}/{self.num_clients}")
            
            client_state = self.train_client(
                client_id, 
                client_datasets[client_id],
                val_data,
                local_epochs
            )
            client_states.append(client_state)

        # Aggregate models with differential privacy
        if verbose:
            print("Aggregating client models" + 
                  (" with differential privacy" if self.privacy_enabled else ""))
            
        aggregated_state = self.aggregate_models(client_states)
        self.model.load_state_dict(aggregated_state)

        # Evaluate global model on test data
        if verbose:
            print("Evaluating global model on test data")
            
        accuracy = self.evaluate(test_data)
        
        # Calculate privacy loss if privacy is enabled
        privacy_loss = 0
        if self.privacy_enabled:
            privacy_loss = self.calculate_privacy_loss(
                self.privacy_budget, 
                self.noise_scale,
                num_queries=len(client_states)
            )
        
        if verbose:
            print(f"Global model accuracy: {accuracy:.2f}%")
            if self.privacy_enabled:
                print(f"Privacy loss: {privacy_loss:.4f}")

        return accuracy, privacy_loss

    def evaluate(self, test_data):
        """
        Evaluate global model on test data
        
        Args:
            test_data (Dataset): The test dataset
            
        Returns:
            float: Accuracy of the model on the test data
        """
        self.model.eval()
        self.model = self.model.to(self.device)
        
        correct = 0
        total = 0
        test_loss = 0

        dataloader = DataLoader(test_data, batch_size=self.config['batch_size'])

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss = test_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        if self.config.get('verbose', True):
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {accuracy:.2f}%")
        
        return accuracy



    def add_differential_privacy_noise(self, tensor, noise_scale):
        """
        Add Gaussian noise to a tensor for differential privacy
        
        Args:
            tensor (torch.Tensor): The tensor to add noise to
            noise_scale (float): The scale of the noise
            
        Returns:
            torch.Tensor: The noisy tensor
        """
        if noise_scale <= 0:
            return tensor
            
        # Generate noise with the same shape as the tensor
        noise = torch.randn_like(tensor) * noise_scale
        
        # Add noise to the tensor
        noisy_tensor = tensor + noise
        
        return noisy_tensor


    def calculate_privacy_loss(self, privacy_budget, noise_scale, num_queries=1):
        """
        Calculate privacy loss for a differentially private mechanism
        
        Args:
            privacy_budget (float): The privacy budget (epsilon)
            noise_scale (float): The noise scale (sigma)
            num_queries (int): Number of queries made
            
        Returns:
            float: The privacy loss
        """
        import numpy as np
        
        if noise_scale <= 0:
            return float('inf')
            
        # This is a simple privacy loss calculation model
        # Real-world applications might use more sophisticated accounting methods
        # such as Moments Accountant or Renyi Differential Privacy
        
        # For Gaussian mechanism, privacy loss can be approximated as:
        # ε ≈ num_queries * (privacy_budget * (1 - exp(-1/noise_scale)))
        
        privacy_loss = num_queries * (privacy_budget * (1 - np.exp(-1/noise_scale)))
        
        return privacy_loss

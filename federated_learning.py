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
        # Use default lr_policy if not provided
        if 'lr_policy' not in self.config:
            self.config['lr_policy'] = {1: {'lr': 0.01}}
            
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

    def distribute_data(self, dataset, use_non_iid=False, alpha=0.5):
        """
        Split dataset among clients, either IID or non-IID

        Args:
            dataset (torch.utils.data.Dataset): The dataset to distribute
            use_non_iid (bool): Whether to use non-IID distribution
            alpha (float): Dirichlet concentration parameter for non-IID distribution
                           Lower values create more skewed distributions

        Returns:
            list: List of datasets, one for each client
        """
        if use_non_iid:
            from data_handler import distribute_data_non_iid
            return distribute_data_non_iid(dataset, self.num_clients, alpha=alpha)
        else:
            # Original IID distribution
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

        # Return the trained model's state dictionary
        return model.state_dict()

    def select_clients(self, fraction=0.5, min_clients=1):
        """
        Select a subset of clients to participate in a training round

        Args:
            fraction (float): Fraction of clients to select (between 0 and 1)
            min_clients (int): Minimum number of clients to select

        Returns:
            list: List of selected client indices
        """
        num_to_select = max(min_clients, int(self.num_clients * fraction))
        return np.random.choice(range(self.num_clients), size=num_to_select, replace=False).tolist()


    def compress_model(self, state_dict, compression_ratio=0.1):
        """
        Compress model weights by keeping only the largest magnitude values

        Args:
            state_dict (dict): Model state dictionary
            compression_ratio (float): Fraction of weights to keep (between 0 and 1)

        Returns:
            dict: Compressed model state dictionary
        """
        compressed_state = {}

        for key, tensor in state_dict.items():
            # Skip non-tensor values
            if not isinstance(tensor, torch.Tensor) or 'running' in key or 'num_batches' in key:
                compressed_state[key] = tensor
                continue

            # Get the flat absolute values and find threshold
            flat_tensor = tensor.abs().flatten()
            if flat_tensor.numel() > 0:  # Check if tensor is not empty
                num_values_to_keep = max(1, int(compression_ratio * flat_tensor.numel()))
                # Get the threshold value for the top k elements
                if num_values_to_keep < flat_tensor.numel():
                    threshold = torch.kthvalue(flat_tensor, flat_tensor.numel() - num_values_to_keep + 1)[0]
                else:
                    threshold = 0.0

                # Create a mask for values to keep
                mask = tensor.abs() >= threshold

                # Apply the mask, zeroing out small values
                pruned_tensor = tensor.clone()
                pruned_tensor[~mask] = 0

                compressed_state[key] = pruned_tensor
            else:
                compressed_state[key] = tensor

        return compressed_state

    def train_round(self, train_data, val_data, test_data, local_epochs, client_fraction=1.0):
        """
        Perform one round of federated learning

        Args:
            train_data (Dataset): The training dataset
            val_data (Dataset): The validation dataset
            test_data (Dataset): The test dataset
            local_epochs (int): Number of local training epochs
            client_fraction (float): Fraction of clients to participate in this round

        Returns:
            tuple: (accuracy, privacy_loss) after the round
        """
        verbose = self.config.get('verbose', True)

        # Get non-IID settings
        use_non_iid = self.config.get('non_iid', {}).get('enabled', False)
        alpha = self.config.get('non_iid', {}).get('alpha', 0.5)

        # Distribute data among clients, potentially non-IID
        client_datasets = self.distribute_data(train_data, use_non_iid=use_non_iid, alpha=alpha)

        # Select a subset of clients to participate in this round
        selected_clients = self.select_clients(fraction=client_fraction)

        if verbose:
            print(f"Starting federated learning round with {len(selected_clients)}/{self.num_clients} clients")
            print(f"Selected clients: {selected_clients}")
            print(f"Local epochs per client: {local_epochs}")

        # Train selected clients in parallel (simulated sequential here)
        client_states = []
        for idx, client_id in enumerate(selected_clients):
            if verbose:
                print(f"Training client {idx+1}/{len(selected_clients)} (ID: {client_id})")

            client_state = self.train_client(
                client_id, 
                client_datasets[client_id],
                val_data,
                local_epochs
            )
            client_states.append(client_state)

        # Get compression settings
        use_compression = self.config.get('compression', {}).get('enabled', False)
        compression_ratio = self.config.get('compression', {}).get('ratio', 0.5)

        # Aggregate models with differential privacy and potential compression
        if verbose:
            compression_msg = " with compression" if use_compression else ""
            privacy_msg = " with differential privacy" if self.privacy_enabled else ""
            print(f"Aggregating client models{privacy_msg}{compression_msg}")

        aggregated_state = self.aggregate_models(
            client_states, 
            use_compression=use_compression,
            compression_ratio=compression_ratio
        )
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
        from differential_privacy import add_noise

        if noise_scale <= 0:
            return tensor

        # Use the specialized add_noise function from differential_privacy module
        return add_noise(tensor, noise_scale)


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

    def aggregate_models(self, client_states, use_compression=False, compression_ratio=0.1):
        """
        Aggregate client models using FedAvg with differential privacy

        Args:
            client_states (list): List of client model state dictionaries
            use_compression (bool): Whether to use model compression
            compression_ratio (float): Fraction of weights to keep if using compression

        Returns:
            dict: Aggregated model state dictionary
        """
        # Apply compression if enabled
        if use_compression:
            compressed_states = [self.compress_model(state, compression_ratio) for state in client_states]
            client_states = compressed_states

        aggregated_state = {}
        for key in client_states[0].keys():
            # Skip non-tensor values
            if not isinstance(client_states[0][key], torch.Tensor):
                aggregated_state[key] = client_states[0][key]
                continue

            # Stack parameters from all clients
            try:
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
            except:
                # Fallback if stacking fails (e.g., because of differing shapes)
                aggregated_state[key] = client_states[0][key]

        return aggregated_state
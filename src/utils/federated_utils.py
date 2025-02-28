"""
Utility functions and helper classes for federated learning implementation.
Includes optimizer factory, scheduler factory, loss function factory, and other utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class OptimizerFactory:
    """Factory class for creating optimizers"""
    
    @staticmethod
    def create(parameters, optimizer_type, **kwargs):
        """
        Create an optimizer based on the specified type
        
        Args:
            parameters: Model parameters to optimize
            optimizer_type (str): Type of optimizer ('sgd', 'adam', 'rmsprop', etc.)
            **kwargs: Additional optimizer parameters
            
        Returns:
            torch.optim.Optimizer: The created optimizer
        """
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'sgd':
            return optim.SGD(parameters, **kwargs)
        elif optimizer_type == 'adam':
            return optim.Adam(parameters, **kwargs)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(parameters, **kwargs)
        elif optimizer_type == 'adagrad':
            return optim.Adagrad(parameters, **kwargs)
        elif optimizer_type == 'adadelta':
            return optim.Adadelta(parameters, **kwargs)
        else:
            print(f"Warning: Unknown optimizer type '{optimizer_type}'. Using SGD as default.")
            return optim.SGD(parameters, **kwargs)


class SchedulerFactory:
    """Factory class for creating learning rate schedulers"""
    
    @staticmethod
    def create(optimizer, scheduler_type, params=None):
        """
        Create a learning rate scheduler based on the specified type
        
        Args:
            optimizer: The optimizer to schedule
            scheduler_type (str): Type of scheduler ('policy', 'step', 'exponential', 'cosine', etc.)
            params (dict): Parameters for the scheduler
            
        Returns:
            torch.optim.lr_scheduler._LRScheduler or None: The created scheduler
        """
        if params is None:
            params = {}
            
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == 'none' or scheduler_type is None:
            return None
        elif scheduler_type == 'policy':
            # This is handled manually in the client training method
            return None
        elif scheduler_type == 'step':
            step_size = params.get('step_size', 2)
            gamma = params.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'multistep':
            milestones = params.get('milestones', [2, 5, 8])
            gamma = params.get('gamma', 0.1)
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif scheduler_type == 'exponential':
            gamma = params.get('gamma', 0.9)
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_type == 'cosine':
            T_max = params.get('T_max', 8)
            eta_min = params.get('eta_min', 0)
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == 'reduce_on_plateau':
            mode = params.get('mode', 'min')
            factor = params.get('factor', 0.1)
            patience = params.get('patience', 2)
            threshold = params.get('threshold', 1e-4)
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, 
                patience=patience, threshold=threshold
            )
        else:
            print(f"Warning: Unknown scheduler type '{scheduler_type}'. No scheduler will be used.")
            return None


class LossFactory:
    """Factory class for creating loss functions"""
    
    @staticmethod
    def create(loss_type, **kwargs):
        """
        Create a loss function based on the specified type
        
        Args:
            loss_type (str): Type of loss function
            **kwargs: Additional loss function parameters
            
        Returns:
            torch.nn.Module: The created loss function
        """
        loss_type = loss_type.lower()
        
        if loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss(**kwargs)
        elif loss_type == 'bce':
            return nn.BCELoss(**kwargs)
        elif loss_type == 'bce_with_logits':
            return nn.BCEWithLogitsLoss(**kwargs)
        elif loss_type == 'mse':
            return nn.MSELoss(**kwargs)
        elif loss_type == 'l1':
            return nn.L1Loss(**kwargs)
        elif loss_type == 'smooth_l1':
            return nn.SmoothL1Loss(**kwargs)
        elif loss_type == 'kl_div':
            return nn.KLDivLoss(**kwargs)
        elif loss_type == 'nll':
            return nn.NLLLoss(**kwargs)
        else:
            print(f"Warning: Unknown loss type '{loss_type}'. Using CrossEntropyLoss as default.")
            return nn.CrossEntropyLoss(**kwargs)


def apply_lr_policy(optimizer, epoch, lr_policy):
    """
    Apply a learning rate policy to an optimizer
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to update
        epoch (int): Current epoch
        lr_policy (dict): Learning rate policy dictionary
        
    Returns:
        float: The applied learning rate
    """
    # Find the applicable learning rate for the current epoch
    epoch_key = epoch
    if epoch_key not in lr_policy:
        # Find the closest lr policy epoch that's smaller
        available_epochs = sorted([k for k in lr_policy.keys() if k <= epoch_key])
        if available_epochs:
            epoch_key = available_epochs[-1]
        else:
            epoch_key = min(lr_policy.keys())
    
    lr = lr_policy[epoch_key]['lr']
    
    # Update optimizer with current learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr



class DefaultFederatedConfig:
    """Default configuration for federated learning"""
    
    @staticmethod
    def get_config():
        """
        Get the default configuration for federated learning
        
        Returns:
            dict: Default configuration
        """
        return {
            'max_epochs': 8,
            'batch_size': 50,
            'save_dir': 'weights',
            'weight_decay': 1e-2,
            'optimizer': 'sgd',  # Options: 'sgd', 'adam', 'rmsprop', etc.
            'optimizer_params': {},
            'loss': 'cross_entropy',  # Options: 'cross_entropy', 'bce', 'mse', etc.
            'loss_params': {},
            'lr_scheduler': 'policy',  # Options: 'policy', 'step', 'exponential', 'cosine', 'none'
            'lr_scheduler_params': {
                'step': {'step_size': 2, 'gamma': 0.1},
                'multistep': {'milestones': [2, 5, 8], 'gamma': 0.1},
                'exponential': {'gamma': 0.9},
                'cosine': {'T_max': 8, 'eta_min': 1e-4},
                'reduce_on_plateau': {'mode': 'min', 'factor': 0.1, 'patience': 2}
            },
            'lr_policy': {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}},
            'privacy': {
                'enabled': True,
                'noise_scale': 0.01,
                'privacy_budget': 1.0
            },
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'verbose': True,
            'seed': 42
        }
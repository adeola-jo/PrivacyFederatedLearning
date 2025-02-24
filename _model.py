import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from data_handler import load_mnist_data

class MNISTCNN(nn.Module):
    """
    A CNN model for MNIST classification using PyTorch, which supports GPU acceleration.
    
    Architecture mimics the original:
    conv1 -> pool1 -> relu1 -> conv2 -> pool2 -> relu2 -> flatten -> fc3 -> relu3 -> logits
    """
    def __init__(self, weight_decay=1e-2):
        """
        Initialize the CNN model with the specified architecture.
        
        Args:
            weight_decay: Weight decay factor for L2 regularization
        """
        super(MNISTCNN, self).__init__()
        
        self.weight_decay = weight_decay
        
        # conv1: 5x5 conv, 16 filters, stride 1, pad 0 (VALID padding in original)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        # pool1: 2x2 max pool, stride 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        
        # conv2: 5x5 conv, 32 filters, stride 1, pad 0 (VALID padding in original)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        # pool2: 2x2 max pool, stride 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        
        # Flatten
        self.flatten = nn.Flatten()
        
        # fc3: fully-connected, 512 units
        self.fc3 = nn.Linear(32 * 4 * 4, 512)
        self.relu3 = nn.ReLU()
        
        # logits: fully-connected, 10 units
        self.logits = nn.Linear(512, 10)
        
        # Store layer names for visualization
        self.conv1.name = "conv1"
        self.conv2.name = "conv2"
    
    def forward(self, x):
        """Forward pass through the network"""
        # First convolutional block
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        
        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.logits(x)
        
        return x


def dense_to_one_hot(y, class_count):
    """Convert class indices to one-hot encoded tensors using PyTorch"""
    return torch.eye(class_count)[y]


def draw_conv_filters(epoch, step, layer, save_dir):
    """
    Draw convolutional filters for visualization.
    Adapts the numpy implementation to PyTorch.
    """
    # Get the weights from the layer
    weights = layer.weight.data.clone().cpu()
    
    # Normalize the weights for visualization
    weights_min, weights_max = weights.min(), weights.max()
    weights = (weights - weights_min) / (weights_max - weights_min)
    
    # Create a grid of images
    grid = vutils.make_grid(weights, normalize=False, padding=1)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the grid as an image
    filename = f'{layer.name}_epoch_{epoch:02d}_step_{step:06d}.png'
    vutils.save_image(grid, os.path.join(save_dir, filename))


def train(train_loader, valid_loader, model, criterion, config):
    """
    Train the model, mimicking the original nn.py implementation but using PyTorch.
    
    Args:
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        model: The neural network model
        criterion: Loss function
        config: Dictionary with training configuration
    """
    # Set up device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    lr_policy = config['lr_policy']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    save_dir = config['save_dir']
    weight_decay = config['weight_decay']
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, max_epochs + 1):
        # Get learning rate for this epoch
        if epoch in lr_policy:
            lr = lr_policy[epoch]['lr']
        else:
            lr = lr_policy[max(lr_policy.keys())]['lr']
        
        # Create optimizer with current learning rate
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Training phase
        model.train()
        train_loss = 0.0
        cnt_correct = 0
        num_examples = 0
        step = 0
        
        for batch_x, batch_y in train_loader:
            # Move data to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_size = batch_x.size(0)
            num_examples += batch_size
            
            # Convert one-hot encoded targets to class indices for loss calculation
            if batch_y.dim() > 1:
                batch_y_indices = batch_y.argmax(dim=1)
            else:
                batch_y_indices = batch_y
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(batch_x)
            loss_val = criterion(logits, batch_y_indices)
            
            # Backward pass and optimization
            loss_val.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss_val.item() * batch_size
            
            # Compute classification accuracy
            _, predicted = logits.max(1)
            if batch_y.dim() > 1:
                true_labels = batch_y.argmax(dim=1)
            else:
                true_labels = batch_y
            cnt_correct += (predicted == true_labels).sum().item()
            
            # Print progress
            if step % 5 == 0:
                print(f"epoch {epoch}, step {step*batch_size}/{len(train_loader.dataset)}, batch loss = {loss_val.item():.2f}")
            
            # Visualize filters
            if step % 100 == 0:
                draw_conv_filters(epoch, step*batch_size, model.conv1, save_dir)
            
            # Print intermediate accuracy
            if step > 0 and step % 50 == 0:
                print(f"Train accuracy = {cnt_correct / num_examples * 100:.2f}%")
            
            step += 1
        
        # Print final training accuracy
        train_acc = cnt_correct / num_examples * 100
        train_loss = train_loss / num_examples
        print(f"Train accuracy = {train_acc:.2f}%")
        print(f"Train loss = {train_loss:.4f}")
        
        # Evaluate on validation set
        evaluate("Validation", valid_loader, model, criterion)
    
    return model


def evaluate(name, data_loader, model, criterion):
    """
    Evaluate the model on the given data, mimicking the original nn.py implementation.
    
    Args:
        name: Name of the dataset (e.g., 'Test')
        data_loader: DataLoader for the dataset
        model: The neural network model
        criterion: Loss function
    """
    print(f"\nRunning evaluation: {name}")
    
    device = next(model.parameters()).device
    
    model.eval()
    total_loss = 0.0
    cnt_correct = 0
    num_examples = 0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            # Move data to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_size = batch_x.size(0)
            num_examples += batch_size
            
            # Convert one-hot encoded targets to class indices for loss calculation
            if batch_y.dim() > 1:
                batch_y_indices = batch_y.argmax(dim=1)
            else:
                batch_y_indices = batch_y
            
            # Forward pass
            logits = model(batch_x)
            loss_val = criterion(logits, batch_y_indices)
            
            # Update statistics
            total_loss += loss_val.item() * batch_size
            
            # Compute accuracy
            _, predicted = logits.max(1)
            if batch_y.dim() > 1:
                true_labels = batch_y.argmax(dim=1)
            else:
                true_labels = batch_y
            cnt_correct += (predicted == true_labels).sum().item()
    
    accuracy = cnt_correct / num_examples * 100
    avg_loss = total_loss / num_examples
    
    print(f"{name} accuracy = {accuracy:.2f}%")
    print(f"{name} avg loss = {avg_loss:.4f}\n")
    
    return accuracy, avg_loss


def main(data_dir='./data', save_dir='./model_output'):
    """
    Main function to train and evaluate the model.
    
    Args:
        data_dir: Directory to store the dataset
        save_dir: Directory to save model checkpoints
    """
    # Create config dictionary
    config = {
        'max_epochs': 8,
        'batch_size': 50,
        'save_dir': save_dir,
        'weight_decay': 1e-2,
        'lr_policy': {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}}
    }
    
    # Load and preprocess data
    train_loader, val_loader, test_loader = load_mnist_data(
        data_dir=data_dir, 
        batch_size=config['batch_size']
    )
    
    # Create model
    model = MNISTCNN(weight_decay=config['weight_decay'])
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    model = train(train_loader, val_loader, model, criterion, config)
    
    # Evaluate on test set
    evaluate("Test", test_loader, model, criterion)


if __name__ == "__main__":
    main()
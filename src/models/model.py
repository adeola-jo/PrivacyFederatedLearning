"""
Neural network model architecture module.
Defines the convolutional neural network used for federated learning on MNIST data.
"""

#TODO: Port the architecture you designed in zagreb to this place and use the network here

import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn
import torch.nn.functional as F

class SimpleConvNet(nn.Module):
    def __init__(self, weight_decay=1e-2):
        """
        A CNN model for MNIST classification using PyTorch, which supports GPU acceleration.
        
        Architecture mimics the original:
        conv1 -> pool1 -> relu1 -> conv2 -> pool2 -> relu2 -> flatten -> fc3 -> relu3 -> logits
        """
        
        super(SimpleConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(32 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        
        # Store layer names for visualization (like in MNISTCNN)
        self.conv1.name = "conv1"
        self.conv2.name = "conv2"
    
    def forward(self, x):
        # First convolutional block - using functional API
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # Second convolutional block - using functional API
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        
        # Flatten and fully connected layers
        x = x.view(-1, 32 * 4 * 4)  # Same flattened size as MNISTCNN
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)  





# ---------------------- OLD CODE --------------------------------
# class SimpleConvNet(nn.Module):
#     def __init__(self):
#         super(SimpleConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
    
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


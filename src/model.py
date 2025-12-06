"""CNN models for microstructure classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MicrostructureCNN(nn.Module):
    """
    Convolutional Neural Network for classifying microstructure images.
    
    Architecture:
    - 4 convolutional blocks with batch normalization and max pooling
    - 2 fully connected layers with dropout
    - Output layer for 6-class classification
    
    Input: (batch_size, 1, 64, 64) grayscale images
    Output: (batch_size, 6) class logits
    """
    
    def __init__(self, num_classes=6, dropout_rate=0.5):
        """
        Initialize the CNN model.
        
        Parameters
        ----------
        num_classes : int
            Number of output classes (default: 6)
        dropout_rate : float
            Dropout probability for regularization (default: 0.5)
        """
        super(MicrostructureCNN, self).__init__()
        
        # Convolutional Block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Convolutional Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolutional Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Convolutional Block 4: 128 -> 256 channels
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        # Input: 64x64
        # After pool1: 32x32
        # After pool2: 16x16
        # After pool3: 8x8
        # After pool4: 4x4
        # Flattened: 256 * 4 * 4 = 4096
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 64, 64)
            
        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes)
        """
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 64x64 -> 32x32
        
        # Block 2: Conv -> BN -> ReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Block 3: Conv -> BN -> ReLU -> Pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Block 4: Conv -> BN -> ReLU -> Pool
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


class MicrostructureDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for microstructure images.
    """
    
    def __init__(self, X, y, transform=None):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        X : np.ndarray
            Input images, shape (n_samples, 64, 64) or (n_samples, 64*64)
        y : np.ndarray
            Labels, shape (n_samples,)
        transform : callable, optional
            Optional transform to apply to images
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
        
        # Reshape if flattened
        if len(self.X.shape) == 2:
            n_samples = self.X.shape[0]
            self.X = self.X.view(n_samples, 1, 64, 64)
        elif len(self.X.shape) == 3:
            # Add channel dimension
            self.X = self.X.unsqueeze(1)
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the sample
            
        Returns
        -------
        tuple
            (image, label) where image is shape (1, 64, 64)
        """
        image = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_model_summary(model, input_size=(1, 1, 64, 64)):
    """
    Print a summary of the model architecture.
    
    Parameters
    ----------
    model : nn.Module
        The model to summarize
    input_size : tuple
        Size of input tensor (batch_size, channels, height, width)
    """
    device = next(model.parameters()).device
    x = torch.randn(*input_size).to(device)
    
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input shape: {input_size}")
    
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")
    print("=" * 70)

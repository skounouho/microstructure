#!/usr/bin/env python3
"""Tools for analyzing VTI grain microstructure data."""

import argparse
from datetime import datetime
from pathlib import Path
from tkinter import Y

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import joblib

from src.utils import load_vti, get_xy_slice
from src.model import MicrostructureCNN, MicrostructureDataset, get_model_summary


DATA_DIR = Path(__file__).parent / "data"
LOGS_DIR = Path(__file__).parent / "logs"
MODELS_DIR = Path(__file__).parent / "models"

RANDOM_SEED = 42

# Available VTI files
VTI_FILES = {
    "r1_P22": "r1_P22_grainid_rgbz.vti",
    "r1_P61": "r1_P61_grainid_rgbz.vti",
    "r1_P63": "r1_P63_grainid_rgbz.vti",
    "r2_P22": "r2_P22_grainid_rgbz.vti",
    "r2_P61": "r2_P61_grainid_rgbz.vti",
    "r2_P63": "r2_P63_grainid_rgbz.vti",
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tools for analyzing VTI grain microstructure data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        required=True,
    )
    
    # Visualize subcommand
    viz_parser = subparsers.add_parser(
        'visualize',
        aliases=['viz', 'plot'],
        help='Visualize XY slices of VTI data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available datasets: {', '.join(VTI_FILES.keys())} or npz file",
    )
    viz_parser.add_argument(
        "dataset",
        # choices=list(VTI_FILES.keys()),
        help="Name of the dataset to visualize",
    )
    viz_parser.add_argument(
        "-z", "--z-index",
        type=int,
        default=None,
        help="Z slice index to display (default: middle slice)",
    )
    viz_parser.set_defaults(func=visualize)

    # Dataset subcommand
    dataset_parser = subparsers.add_parser(
        'data',
        help='Create a dataset from the VTI data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dataset_parser.add_argument(
        "--defects",
        type=int,
        default=0,
        help="Add defects to the dataset",
    )
    dataset_parser.set_defaults(func=dataset)

    # Train subcommand
    train_parser = subparsers.add_parser(
        'train',
        help='Train a machine learning model on the dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_parser.add_argument(
        "--type",
        choices=['cnn', 'svc'],
        default='svc',
        help="Type of model to train",
    )
    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser(
        'test',
        help='Test a machine learning model on the dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    test_parser.add_argument(
        "--type",
        choices=['cnn', 'svc'],
        default='svc',
        help="Type of model to test",
    )
    test_parser.add_argument(
        "--datafile",
        help="Data file to test the model on",
    )
    test_parser.set_defaults(func=test)
    
    return parser.parse_args()

def visualize(args):
    """Visualize the XY slice of the VTI file."""

    # Load the VTI file or data.npz file
    if args.dataset.endswith('npz'):
        data = np.load(DATA_DIR / args.dataset)
        X = data["X"].reshape(-1, 64, 64).transpose(1, 2, 0)  # shape: (n_samples, 64, 64)
        gray = X
        dims = X.shape
        categories = data["y"].squeeze()
        # spacing = (1.0, 1.0, 1.0)
    else:
        vti_path = DATA_DIR / VTI_FILES[args.dataset]
        print(f"Loading {vti_path}...")
        vti_data = load_vti(str(vti_path))
    
        gray = vti_data['gray']
        dims = vti_data['dimensions']
        # spacing = vti_data['spacing']
    
    print(f"Data dimensions: {dims[0]} x {dims[1]} x {dims[2]}")
    # print(f"Voxel spacing: {spacing}")
    
    # Determine Z slice index
    z_index = args.z_index if args.z_index is not None else dims[2] // 2
    
    # Validate z_index
    if z_index < 0 or z_index >= dims[2]:
        print(f"Error: z_index {z_index} out of range [0, {dims[2] - 1}]")
        return 1
    
    # Extract the XY slice of gray data
    slice_gray = get_xy_slice(gray, z_index)  # shape: (nx, ny, 1)
    print(f"Displaying XY slice at Z={z_index}")
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display the RGB slice with origin at lower-left
    # Transpose to get correct orientation (X horizontal, Y vertical)
    # Need to swap axes 0 and 1 while keeping RGB channel last
    slice_display = np.transpose(slice_gray, (1, 0))

    print(slice_display.min(), slice_display.max())
    
    ax.imshow(
        slice_display,
        origin='lower',
        aspect='equal',
        cmap='grey',
        vmin=0,
        vmax=1,
    )
    
    # Labels and title
    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')
    ax.set_title(f'{args.dataset} — XY Slice at Z={z_index} (Gray scale), Categories: {categories[z_index]}')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return 0

def dataset(args):
    """Train a machine learning model on the VTI data."""

    if args.defects > 0:
        defect_dataset(args)
        return 0

    data = {}

    for key, vti_file in VTI_FILES.items():
        vti_path = DATA_DIR / vti_file
        print(f"Loading {vti_path}...")
        vti_data = load_vti(str(vti_path))
        data[key] = vti_data['gray']
        print(f"Dimensions: {data[key].shape}")

    # organize data into input array and output array
    # input array is a 4D array of shape (n_samples, n_features, n_x, n_y)
    # where n_samples is the number of Z slices, n_features is 3 for the RGB values
    # output array is a 2D array of shape (n_samples, 1)
    # where the value is a integer corresponding to the key of the VTI file

    print("Formatting data...")

    X = []
    y = []
    for i, key in enumerate(data.keys()):
        raw_data = data[key]
        downsampled_data = raw_data[::raw_data.shape[0]//65, ::raw_data.shape[1]//65, :]
        X.append(downsampled_data[:64, :64, :]) # shape: (n_x, n_y, n_z, 1)
        y.append(np.repeat(i, downsampled_data.shape[2])) # shape: (n_z,)
    
    X = np.concatenate(X, axis=2).transpose(2, 3, 0, 1) # shape: (n_samples, n_x, n_y, 1)
    y = np.concatenate(y, axis=0) # shape: (n_samples,)

    print(X.shape)
    print(y.shape)

    X = X.squeeze()
    n_samples = X.shape[0]
    X = X.reshape(n_samples, -1)
    y = y.reshape(n_samples, 1)

    print(X.shape)
    print(y.shape)

    np.savez_compressed(DATA_DIR / "data.npz", X=X, y=y)
    print(f"Cached data to {DATA_DIR / 'data.npz'}")

    return 0

def defect_dataset(args):
    """Create a dataset with defects."""
    
    data = np.load(DATA_DIR / "data.npz")

    X = data["X"].reshape(-1, 64, 64)
    y = data["y"]

    # add defects
    generator = np.random.default_rng(RANDOM_SEED)
    for i in range(len(X)):
        X[i] = add_defects(X[i], generator=generator, num_defects=args.defects)
    
    X = X.reshape(-1, 64 * 64)

    np.savez_compressed(DATA_DIR / f"defect_data_{args.defects}.npz", X=X, y=y)
    print(f"Cached data to {DATA_DIR / f'defect_data_{args.defects}.npz'}")

    return 0

def add_defects(arr: np.ndarray, generator: np.random.Generator, num_defects: int = 10) -> np.ndarray:
    """Add defects to the array."""
    radius = 2
    xpos, ypos = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
    for _ in range(num_defects):
        x = generator.integers(0, arr.shape[0])
        y = generator.integers(0, arr.shape[1])
        mask = (xpos - x)**2 + (ypos - y)**2 <= radius**2
        arr[mask] = 0.0
    return arr

def train(args):
    """Train a machine learning model on the dataset."""
    if args.type == 'cnn':
        train_cnn(args)
    elif args.type == 'svc':
        train_svc(args)
    else:
        print(f"Invalid model type: {args.type}")
        return 1
    return 0


def train_cnn(args):
    """Train a machine learning model on the dataset."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading dataset...")
    data = np.load(DATA_DIR / "data.npz")
    X = data["X"].reshape(-1, 64, 64)  # shape: (n_samples, 64, 64)
    y = data["y"].squeeze()  # shape: (n_samples,)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create dataset
    dataset = MicrostructureDataset(X, y)
    
    # Split into train/val sets
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = MicrostructureCNN(num_classes=6)
    model = model.to(device)
    get_model_summary(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training parameters
    num_epochs = 50
    best_val_loss = float('inf')
    best_model_path = MODELS_DIR / "cnn.pth"
    
    # Create logs directory and CSV log file
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"training_{timestamp}.csv"
    
    # Write CSV header with setup parameters as comments
    with open(log_file, 'w') as f:
        f.write(f"# Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Device: {device}\n")
        f.write(f"# Data shape: X={X.shape}, y={y.shape}\n")
        f.write(f"# Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}\n")
        f.write(f"# Batch size: {batch_size}\n")
        f.write(f"# Number of epochs: {num_epochs}\n")
        f.write(f"# Learning rate: 0.001\n")
        f.write(f"# Optimizer: Adam\n")
        f.write(f"# Loss function: CrossEntropyLoss\n")
        f.write("#\n")
        # CSV header row
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
    
    print(f"\nLogging to: {log_file}")
    print(f"Starting training for {num_epochs} epochs...")
    print("=" * 70)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Log to CSV file
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{train_acc:.4f},{avg_val_loss:.6f},{val_acc:.4f}\n")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': val_acc,
            }, best_model_path)
            print(f"  → Saved best model (val_loss: {best_val_loss:.4f})")
    
    print("=" * 70)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {best_model_path}")
    print(f"Training log saved to: {log_file}")
    
    return 0


def train_svc(args):
    """Train a clustering model on the dataset."""

    # Load data
    print("Loading dataset...")
    data = np.load(DATA_DIR / "data.npz")
    X = data["X"].reshape(-1, 64 * 64)  # shape: (n_samples, 64 * 64)
    y = data["y"].squeeze()  # shape: (n_samples,)
    print(f"Data shape: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit using support-vector machine classification with scikit
    clf = SVC(C=1, kernel='linear')
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_test, y_test)

    print(f"Train Acc: {train_acc:.2%} | "
          f"Val Acc: {val_acc:.2%}")

    # Save the model
    joblib.dump(clf, MODELS_DIR / "svc.pkl")
    print(f"Saved model to {MODELS_DIR / 'svc.pkl'}")

    return 0

def test(args):
    """Test a machine learning model on the dataset."""
    if args.type == 'cnn':
        test_cnn(args)
    elif args.type == 'svc':
        test_svc(args)
    else:
        print(f"Invalid model type: {args.type}")
        return 1
    return 0

def test_cnn(args):
    """Test a CNN model on the dataset."""
    model = MicrostructureCNN(num_classes=6)
    model.load_state_dict(torch.load(MODELS_DIR / "cnn.pth"))
    model.eval()

    datafile = args.datafile
    data = np.load(DATA_DIR / datafile)
    X = data["X"].reshape(-1, 64, 64)
    y = data["y"].squeeze()

    with torch.no_grad():
        outputs = model(X)
    _, predicted = torch.max(outputs.data, 1)
    test_acc = (predicted == y).sum().item() / len(y)
    print(f"Acc: {test_acc:.2%}")
    
    return 0

def test_svc(args):
    """Test a SVC model on the dataset."""
    model = joblib.load(MODELS_DIR / "svc.pkl")

    datafile = args.datafile
    data = np.load(DATA_DIR / datafile)
    X = data["X"]
    y = data["y"].squeeze()

    predicted = model.predict(X)
    test_acc = (predicted == y).sum().item() / len(y)
    print(f"Acc: {test_acc:.2%}")

    return 0

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Call the appropriate subcommand function
    return args.func(args) or 0


if __name__ == "__main__":
    exit(main())

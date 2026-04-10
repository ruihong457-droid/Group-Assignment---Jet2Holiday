"""
Utility functions module
"""
import os
import random
import yaml
import torch
import numpy as np
from torchvision import datasets


def load_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Return the available device (GPU or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_training_env():
    """Configure CUDA/cuDNN for maximum throughput on fixed-shape workloads.

    Must be called AFTER set_seed() — set_seed() resets cudnn.benchmark to False,
    and this function overrides it back to True.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # deterministic must be False when benchmark=True;
        # results remain numerically consistent but not bit-exact across runs
        torch.backends.cudnn.deterministic = False


def get_cifar100_classes():
    """Return the list of CIFAR-100 class names."""
    dataset = datasets.CIFAR100(root='./data', train=False, download=False)
    return dataset.classes


def save_checkpoint(model, optimizer, epoch, val_acc, save_path):
    """Save a model checkpoint to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, save_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load a model checkpoint from disk."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_acc']

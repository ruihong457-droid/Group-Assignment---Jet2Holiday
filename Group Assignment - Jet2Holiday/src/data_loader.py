"""
CIFAR-100 data loading module
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_cifar100_loaders(config):
    """
    Build train, validation, and test DataLoaders for CIFAR-100.

    The official training set (50,000 images) is split 90/10 into train and val.
    The official test set (10,000 images) is kept intact for final evaluation.

    Args:
        config: Dict containing batch_size, num_workers, data_dir, seed, etc.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Data augmentation for the training set
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])

    # Normalisation only for validation and test sets
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])

    # Load the full official training set (data already downloaded, no network request)
    full_train_dataset = datasets.CIFAR100(
        root=config.get('data_dir', './data'),
        train=True,
        download=False,
        transform=train_transform
    )

    # Split into train (45,000) and val (5,000)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )

    # Replace the val subset's underlying dataset with a clean-transform version
    # so validation images are never augmented
    val_dataset.dataset = datasets.CIFAR100(
        root=config.get('data_dir', './data'),
        train=True,
        download=False,
        transform=test_transform
    )

    # Load the official test set
    test_dataset = datasets.CIFAR100(
        root=config.get('data_dir', './data'),
        train=False,
        download=False,
        transform=test_transform
    )

    # Build DataLoaders
    _persistent = config.get('num_workers', 4) > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 128),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=_persistent,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 128),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=_persistent,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 128),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=_persistent,
    )

    return train_loader, val_loader, test_loader


def get_class_names():
    """Return the list of CIFAR-100 class names."""
    dataset = datasets.CIFAR100(root='./data', train=False, download=False)
    return dataset.classes

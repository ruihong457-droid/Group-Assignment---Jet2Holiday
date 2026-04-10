"""
Model definition module
"""
import torch.nn as nn
from torchvision import models


def get_resnet18(num_classes=100, pretrained=True, freeze_features=False):
    """
    Build a ResNet18 model with the final FC layer replaced for the target number of classes.

    Args:
        num_classes: Number of output classes (default 100 for CIFAR-100)
        pretrained: Whether to load ImageNet pre-trained weights (default True)
        freeze_features: Whether to freeze all layers except the FC head (default False)

    Returns:
        model: Modified ResNet18 model
    """
    # Load ResNet18 using the new weights API (torchvision >= 0.13)
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Optionally freeze the feature extractor layers
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final FC layer to match the target number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

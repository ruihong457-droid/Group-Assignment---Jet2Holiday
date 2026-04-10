"""
Single experiment runner
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path so src/ imports work regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.utils import load_config, set_seed, get_device, setup_training_env
from src.data_loader import get_cifar100_loaders
from src.model import get_resnet18
from src.trainer import Trainer
from src.evaluator import Evaluator


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.

    Soft labels replace hard one-hot targets: the correct class receives
    (1 - smoothing) probability, with the remainder distributed uniformly
    across the other classes. Reduces overconfident predictions.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_prob = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            smooth_target = torch.full_like(log_prob, self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return (-smooth_target * log_prob).sum(dim=-1).mean()


def run_experiment(config_path=None, config=None):
    """Run a single experiment.

    Args:
        config_path: Path to a YAML config file (mutually exclusive with config)
        config: Config dict (used by run_all_experiments.py to pass modified configs)
    """
    if config is None:
        if config_path is None:
            raise ValueError('Provide either config_path or config')
        config = load_config(config_path)

    # Reproducibility
    set_seed(config['training']['seed'])

    # Device + CUDA optimisations (called after set_seed to override cudnn settings)
    device = get_device()
    setup_training_env()
    print(f'Using device: {device}')

    # Data
    print('Loading data...')
    train_loader, val_loader, test_loader = get_cifar100_loaders(config['training'])

    # Model
    print('Creating model...')
    model = get_resnet18(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)

    # torch.compile — requires Triton, which is not supported on Windows
    import sys as _sys
    if _sys.platform != 'win32':
        try:
            model = torch.compile(model)
            print('torch.compile enabled')
        except Exception as e:
            print(f'Warning: torch.compile not available ({e}), continuing without it')
    else:
        print('torch.compile skipped (Triton not supported on Windows)')

    # Loss function
    loss_type = config['loss']['type']
    if loss_type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'LabelSmoothingCrossEntropy':
        smoothing = config['loss'].get('smoothing', 0.1)
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        raise ValueError(f'Unsupported loss type: {loss_type}')

    # Optimiser — created after torch.compile so it binds to the compiled parameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['optimizer'].get('weight_decay', 0.0001)
    )

    # Training
    print('Starting training...')
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer,
        device, config['early_stopping']
    )
    history = trainer.train(
        num_epochs=config['training']['epochs'],
        save_dir=config['paths']['checkpoint_dir']
    )

    # Load the best checkpoint before test evaluation
    best_model_path = os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f'Loaded best model from {best_model_path}')
    else:
        print('Warning: best_model.pth not found, using last epoch weights for evaluation')

    # Test set evaluation
    print('Evaluating on test set...')
    evaluator = Evaluator(model, test_loader, device)
    results = evaluator.evaluate()

    # Save results
    results['history'] = history
    results['config'] = config

    os.makedirs(config['paths']['result_dir'], exist_ok=True)
    result_path = os.path.join(config['paths']['result_dir'], 'results.json')
    evaluator.save_results(results, result_path)

    print(f'\nResults saved to {result_path}')
    print(f"Test Top-1 Accuracy: {results['top1_acc']:.2f}%")
    print(f"Test Top-5 Accuracy: {results['top5_acc']:.2f}%")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run a single CIFAR-100 experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    run_experiment(config_path=args.config)

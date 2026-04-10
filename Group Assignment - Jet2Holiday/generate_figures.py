"""
Generate all required visualisation figures
"""
import os
import sys
import json
import torch

sys.path.append(os.path.dirname(__file__))

from src.visualizer import (
    plot_training_curves,
    plot_loss_function_comparison,
    plot_comparison_loss,
    plot_comparison_accuracy,
    plot_predictions
)
from src.model import get_resnet18
from src.data_loader import get_cifar100_loaders
from src.utils import get_device


def load_results(result_dir):
    """Load experiment results from a results.json file."""
    result_path = os.path.join(result_dir, 'results.json')
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            return json.load(f)
    return None


def generate_all_figures():
    """Generate all 7 required figures from saved experiment results."""
    os.makedirs('./figures', exist_ok=True)

    # Figure 1: Baseline training curves (loss + accuracy vs epoch)
    print('Generating Figure 1: Baseline Training Curves...')
    baseline_results = load_results('./results/exp1_baseline')
    if baseline_results and 'history' in baseline_results:
        plot_training_curves(
            baseline_results['history'],
            './figures/fig1_baseline_curves.png',
            'Baseline Experiment'
        )

    # Figure 2: Loss function comparison (CrossEntropy vs LabelSmoothing, two-panel)
    print('Generating Figure 2: Loss Function Comparison...')
    loss_results = load_results('./results/exp2_loss')
    if baseline_results and loss_results:
        results_dict = {
            'CrossEntropyLoss': baseline_results['history'],
            'LabelSmoothing (ε=0.1)': loss_results['history']
        }
        plot_loss_function_comparison(
            results_dict,
            './figures/fig2_loss_comparison.png',
            'Loss Function Comparison'
        )

    # Figures 3-4: Learning rate comparison
    print('Generating Figures 3-4: Learning Rate Comparison...')
    lr_results = {}
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        result_dir = f'./results/exp3_lr/lr_{lr}'
        results = load_results(result_dir)
        if results:
            lr_results[f'LR={lr}'] = results['history']

    if lr_results:
        plot_comparison_loss(
            lr_results,
            './figures/fig3_lr_loss.png',
            'Learning Rate Comparison - Loss'
        )
        plot_comparison_accuracy(
            lr_results,
            './figures/fig4_lr_accuracy.png',
            'Learning Rate Comparison - Accuracy'
        )

    # Figures 5-6: Batch size comparison
    print('Generating Figures 5-6: Batch Size Comparison...')
    batch_results = {}
    for batch in [8, 16, 32, 64, 128]:
        result_dir = f'./results/exp4_batch/batch_{batch}'
        results = load_results(result_dir)
        if results:
            batch_results[f'Batch={batch}'] = results['history']

    if batch_results:
        plot_comparison_loss(
            batch_results,
            './figures/fig5_batch_loss.png',
            'Batch Size Comparison - Loss'
        )
        plot_comparison_accuracy(
            batch_results,
            './figures/fig6_batch_accuracy.png',
            'Batch Size Comparison - Accuracy'
        )

    # Figure 7: Prediction visualisation — first 100 test samples in a 10×10 grid
    print('Generating Figure 7: Prediction Visualization...')
    device = get_device()
    config = {'batch_size': 128, 'num_workers': 4, 'seed': 42, 'data_dir': './data'}
    _, _, test_loader = get_cifar100_loaders(config)

    model = get_resnet18().to(device)
    checkpoint_path = './checkpoints/exp1_baseline/best_model.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        plot_predictions(
            model,
            test_loader,
            device,
            './figures/fig7_predictions.png'
        )

    print('\nAll figures generated successfully!')


if __name__ == '__main__':
    generate_all_figures()

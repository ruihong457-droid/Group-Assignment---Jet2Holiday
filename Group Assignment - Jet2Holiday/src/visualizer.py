"""
Visualisation module
"""
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.utils import get_cifar100_classes

# Global plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_training_curves(history, save_path, title='Training Curves'):
    """
    Plot training curves in a two-panel layout.

    Args:
        history: Dict with keys train_loss, train_acc, val_acc
        save_path: File path to save the figure
        title: Figure title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    epochs = range(1, len(history['train_loss']) + 1)

    # Top panel: loss curve
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True)

    # Bottom panel: accuracy curves
    ax2.plot(epochs, history['train_acc'], 'g-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_comparison_loss(results_dict, save_path, title='Loss Comparison'):
    """
    Plot training loss curves for multiple experiments on a single axes.

    Args:
        results_dict: {experiment_name: history} mapping
        save_path: File path to save the figure
        title: Figure title
    """
    plt.figure(figsize=(10, 6))

    for exp_name, history in results_dict.items():
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], label=exp_name, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_comparison_accuracy(results_dict, save_path, title='Accuracy Comparison'):
    """Plot train and val accuracy curves for multiple experiments on a single axes."""
    plt.figure(figsize=(12, 6))
    for exp_name, history in results_dict.items():
        epochs = range(1, len(history['train_acc']) + 1)
        plt.plot(epochs, history['train_acc'], '--', label=f'{exp_name} (Train)', linewidth=1.5)
        plt.plot(epochs, history['val_acc'], '-', label=f'{exp_name} (Val)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_loss_function_comparison(results_dict, save_path, title='Loss Function Comparison'):
    """
    Plot a two-panel comparison figure for two loss function experiments.

    Top panel: training loss curves for each experiment.
    Bottom panel: train and val accuracy curves for each experiment.

    Args:
        results_dict: {experiment_name: history} mapping (train_loss / train_acc / val_acc)
        save_path: File path to save the figure
        title: Overall figure title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    colors = ['blue', 'orange', 'green', 'red']
    for i, (exp_name, history) in enumerate(results_dict.items()):
        epochs = range(1, len(history['train_loss']) + 1)
        color = colors[i % len(colors)]
        ax1.plot(epochs, history['train_loss'], color=color, label=exp_name, linewidth=2)
        ax2.plot(epochs, history['train_acc'], '--', color=color,
                 label=f'{exp_name} (Train)', linewidth=1.5)
        ax2.plot(epochs, history['val_acc'], '-', color=color,
                 label=f'{exp_name} (Val)', linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_predictions(model, test_loader, device, save_path, num_samples=100):
    """Visualise the first num_samples test predictions in a 10×10 grid."""
    import torch
    model.eval()
    classes = get_cifar100_classes()
    images, labels, predictions = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            images.extend(inputs.cpu())
            labels.extend(targets.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            if len(images) >= num_samples:
                break
    images = images[:num_samples]
    labels = labels[:num_samples]
    predictions = predictions[:num_samples]

    # Denormalise using CIFAR-100 channel statistics
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            img = images[idx].numpy().transpose(1, 2, 0)
            img = std * img + mean
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            pred_class = classes[predictions[idx]]
            true_class = classes[labels[idx]]
            # Red text for wrong predictions, green for correct
            color = 'red' if predictions[idx] != labels[idx] else 'green'
            ax.set_title(f'P:{pred_class[:8]}\nT:{true_class[:8]}', fontsize=8, color=color)
            ax.axis('off')
            if predictions[idx] != labels[idx]:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')

print("Script started")
"""
Training module
"""
import os
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config

        # AMP scaler — enabled only on CUDA; gracefully disabled on CPU
        _amp_enabled = (device.type == 'cuda')
        self.scaler = GradScaler(device=device.type, enabled=_amp_enabled)

        # Early stopping state
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.patience = config.get('patience', 10)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def train_epoch(self):
        """Run one training epoch and return (loss, accuracy)."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'loss': f'{running_loss/len(pbar):.3f}',
                            'acc': f'{100.*correct/total:.2f}%'})

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        """Evaluate the model on the validation set and return accuracy."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_acc = 100. * correct / total
        return val_acc

    def train(self, num_epochs, save_dir='./checkpoints'):
        """Run the full training loop with early stopping."""
        print(f'Training on {self.device}')
        print(f'Total epochs: {num_epochs}')

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')

            # Training step
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation step
            val_acc = self.validate()
            self.history['val_acc'].append(val_acc)

            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Acc: {val_acc:.2f}%')

            # Save the best model checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), save_path)
                print(f'Best model saved with val_acc: {val_acc:.2f}%')
            else:
                self.patience_counter += 1

            # Early stopping check
            if self.patience_counter >= self.patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

        return self.history

"""
Evaluation module
"""
import torch
import json
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):
        """Evaluate the model and return Top-1 and Top-5 accuracy."""
        self.model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc='Evaluating'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                # Top-1 accuracy
                _, pred_top1 = outputs.max(1)
                correct_top1 += pred_top1.eq(targets).sum().item()

                # Top-5 accuracy
                _, pred_top5 = outputs.topk(5, 1, True, True)
                pred_top5 = pred_top5.t()
                correct_top5 += pred_top5.eq(targets.view(1, -1).expand_as(pred_top5)).sum().item()

                total += targets.size(0)

                # Collect predictions for downstream visualisation
                all_predictions.extend(pred_top1.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

        top1_acc = 100. * correct_top1 / total
        top5_acc = 100. * correct_top5 / total

        return {
            'top1_acc': top1_acc,
            'top5_acc': top5_acc,
            'predictions': all_predictions,
            'targets': all_targets
        }

    def save_results(self, results, save_path):
        """Save evaluation results to a JSON file."""
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

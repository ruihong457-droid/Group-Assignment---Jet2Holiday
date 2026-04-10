# CIFAR-100 Image Classification — ResNet18 Fine-tuning with Hyperparameter Comparison

Fine-tuning a pre-trained ResNet18 on CIFAR-100 and systematically comparing the effect of loss function, learning rate, and batch size on training dynamics and test accuracy.

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU recommended; tested on RTX 5070)

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn`, `PyYAML`, `tqdm`, `pillow`

---

## Quick Start

### Run a single experiment

```bash
python experiments/run_experiment.py --config config/exp1_baseline.yaml
```

### Run all experiments

```bash
python experiments/run_all_experiments.py
```

Already-completed experiments are skipped automatically. Use `--force` to re-run everything:

```bash
python experiments/run_all_experiments.py --force
```

### Generate figures

```bash
python generate_figures.py
```

Outputs 7 figures to `figures/`.

---

## Project Structure

```
CDS-525/
├── config/
│   ├── base_config.yaml          # Shared defaults
│   ├── exp1_baseline.yaml        # Exp1: baseline settings
│   ├── exp2_loss.yaml            # Exp2: loss function comparison
│   ├── exp3_lr.yaml              # Exp3: learning rate sweep
│   └── exp4_batch.yaml           # Exp4: batch size sweep
├── src/
│   ├── data_loader.py            # CIFAR-100 loading, augmentation, 90/10 split
│   ├── model.py                  # ResNet18 with replaced FC head (→100 classes)
│   ├── trainer.py                # Training loop + early stopping
│   ├── evaluator.py              # Top-1 and Top-5 accuracy
│   ├── visualizer.py             # Figure generation
│   └── utils.py                  # Seed, device, config loading
├── experiments/
│   ├── run_experiment.py         # Single experiment runner + LabelSmoothing loss
│   └── run_all_experiments.py    # Batch runner with skip/force logic
├── generate_figures.py           # Post-training figure generation
├── checkpoints/                  # Saved best_model.pth per experiment
├── results/                      # results.json per experiment
└── figures/                      # Generated plots (fig1–fig7)
```

---

## Experiments

| # | Variable | Settings |
|---|----------|----------|
| Exp1 (Baseline) | — | CrossEntropyLoss, lr=0.001, batch=128, 50 epochs |
| Exp2 (Loss Function) | LabelSmoothing vs CE | Same as Exp1 |
| Exp3 (Learning Rate) | 0.1 / 0.01 / 0.001 / 0.0001 | CE Loss, batch=128, 35 epochs |
| Exp4 (Batch Size) | 8 / 16 / 32 / 64 / 128 | CE Loss, lr=0.001, 35 epochs |

11 experiment variants in total. Exp3 and Exp4 use 35 epochs to keep total runtime reasonable.

### Common settings (all experiments)

- **Model**: ResNet18 (ImageNet pre-trained, full fine-tuning, ~11.23M parameters)
- **Optimiser**: Adam with `weight_decay=1e-4`
- **Early stopping**: patience=10 on validation accuracy
- **Data split**: 45,000 train / 5,000 validation / 10,000 test (fixed seed=42)
- **Augmentation**: RandomCrop(32, padding=4) + RandomHorizontalFlip (train only)

---

## Results

| Experiment | Top-1 Acc | Top-5 Acc |
|-----------|-----------|-----------|
| Exp1 — Baseline (CE, lr=0.001, batch=128) | **57.11%** | **83.67%** |
| Exp2 — LabelSmoothing (smoothing=0.1) | 57.52% | — |
| Exp3 — lr=0.1 | ~5–7% | — |
| Exp3 — lr=0.01 | 34.99% | — |
| Exp3 — lr=0.001 | 57.11% | — |
| Exp3 — lr=0.0001 | ~57% | — |
| Exp4 — batch=128 | 56.40% | — |
| Exp4 — batch=64 | 56.65% | — |
| Exp4 — batch=32 | 55.55% | — |
| Exp4 — batch=16 | 49.57% | — |
| Exp4 — batch=8 | 43.42% | — |

Best single-experiment result: **57.52%** Top-1 (LabelSmoothing).

---

## Output Files

After training, each experiment writes:

- `checkpoints/<exp_name>/best_model.pth` — best checkpoint by validation accuracy
- `results/<exp_name>/results.json` — `top1_acc`, `top5_acc`, and training history

After running `generate_figures.py`:

| File | Content |
|------|---------|
| `figures/fig1_baseline_curves.png` | Baseline: loss + accuracy vs epoch |
| `figures/fig2_loss_comparison.png` | CE vs LabelSmoothing curves |
| `figures/fig3_lr_loss.png` | Loss curves for 4 learning rates |
| `figures/fig4_lr_accuracy.png` | Accuracy curves for 4 learning rates |
| `figures/fig5_batch_loss.png` | Loss curves for 5 batch sizes |
| `figures/fig6_batch_accuracy.png` | Accuracy curves for 5 batch sizes |
| `figures/fig7_predictions.png` | 10×10 grid of first 100 test predictions |


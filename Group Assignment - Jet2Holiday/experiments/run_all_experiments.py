"""
批量运行所有实验（包含多变量对比）
"""
import os
import sys
import copy
import json
import argparse

# 将项目根目录和 experiments/ 目录加入路径（与工作目录无关）
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_script_dir, '..'))
sys.path.insert(0, _project_root)
sys.path.insert(0, _script_dir)

from run_experiment import run_experiment
from src.utils import load_config


def _resolve_paths(config):

    config = copy.deepcopy(config)
    for key in ('checkpoint_dir', 'result_dir'):
        rel = config['paths'][key]
        config['paths'][key] = os.path.normpath(os.path.join(_project_root, rel))

    rel_data = config['training'].get('data_dir', './data')
    config['training']['data_dir'] = os.path.normpath(os.path.join(_project_root, rel_data))
    return config


def _is_completed(result_dir):
    """
    Returns True only if result_dir contains a valid, complete results.json.
    Guards against: missing file, empty file, mid-write truncation, missing keys.
    Any read/parse error returns False (treats as not completed).
    """
    path = os.path.join(result_dir, 'results.json')
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return isinstance(data, dict) and 'top1_acc' in data
    except Exception:
        return False


def run_all_experiments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='Re-run all experiments, ignoring existing results')
    args = parser.parse_args()
    force = args.force

    config_dir = os.path.join(_project_root, 'config')
    results_summary = {}

    print(f'\n{"="*60}\nRunning Exp1: Baseline (CrossEntropyLoss)\n{"="*60}')
    try:
        config = _resolve_paths(load_config(os.path.join(config_dir, 'exp1_baseline.yaml')))
        if not force and _is_completed(config['paths']['result_dir']):
            print('[SKIP] exp1_baseline — already completed (use --force to re-run)')
            results_summary['exp1_baseline'] = {'status': 'skipped'}
        else:
            results = run_experiment(config=config)
            results_summary['exp1_baseline'] = {
                'status': 'success',
                'top1_acc': results['top1_acc'],
                'top5_acc': results['top5_acc']
            }
    except Exception as e:
        print(f'Error: {e}')
        results_summary['exp1_baseline'] = {'status': 'failed', 'error': str(e)}

    print(f'\n{"="*60}\nRunning Exp2: Loss Function (LabelSmoothingCrossEntropy)\n{"="*60}')
    try:
        config = _resolve_paths(load_config(os.path.join(config_dir, 'exp2_loss.yaml')))
        if not force and _is_completed(config['paths']['result_dir']):
            print('[SKIP] exp2_loss — already completed (use --force to re-run)')
            results_summary['exp2_loss'] = {'status': 'skipped'}
        else:
            results = run_experiment(config=config)
            results_summary['exp2_loss'] = {
                'status': 'success',
                'top1_acc': results['top1_acc'],
                'top5_acc': results['top5_acc']
            }
    except Exception as e:
        print(f'Error: {e}')
        results_summary['exp2_loss'] = {'status': 'failed', 'error': str(e)}

    print(f'\n{"="*60}\nRunning Exp3: Learning Rate Variants\n{"="*60}')
    base_exp3 = load_config(os.path.join(config_dir, 'exp3_lr.yaml'))
    for lr in base_exp3['learning_rates']:
        tag = f'lr_{lr}'
        print(f'\n--- {tag} ---')
        try:
            config = copy.deepcopy(base_exp3)
            config['training']['learning_rate'] = lr
            config['paths']['checkpoint_dir'] = f'./checkpoints/exp3_lr/{tag}'
            config['paths']['result_dir'] = f'./results/exp3_lr/{tag}'
            config = _resolve_paths(config)
            if not force and _is_completed(config['paths']['result_dir']):
                print(f'[SKIP] exp3_{tag} — already completed (use --force to re-run)')
                results_summary[f'exp3_{tag}'] = {'status': 'skipped'}
                continue
            results = run_experiment(config=config)
            results_summary[f'exp3_{tag}'] = {
                'status': 'success',
                'top1_acc': results['top1_acc'],
                'top5_acc': results['top5_acc']
            }
        except Exception as e:
            print(f'Error: {e}')
            results_summary[f'exp3_{tag}'] = {'status': 'failed', 'error': str(e)}

    print(f'\n{"="*60}\nRunning Exp4: Batch Size Variants\n{"="*60}')
    base_exp4 = load_config(os.path.join(config_dir, 'exp4_batch.yaml'))
    for batch in base_exp4['batch_sizes']:
        tag = f'batch_{batch}'
        print(f'\n--- {tag} ---')
        try:
            config = copy.deepcopy(base_exp4)
            config['training']['batch_size'] = batch
            config['paths']['checkpoint_dir'] = f'./checkpoints/exp4_batch/{tag}'
            config['paths']['result_dir'] = f'./results/exp4_batch/{tag}'
            config = _resolve_paths(config)
            if not force and _is_completed(config['paths']['result_dir']):
                print(f'[SKIP] exp4_{tag} — already completed (use --force to re-run)')
                results_summary[f'exp4_{tag}'] = {'status': 'skipped'}
                continue
            results = run_experiment(config=config)
            results_summary[f'exp4_{tag}'] = {
                'status': 'success',
                'top1_acc': results['top1_acc'],
                'top5_acc': results['top5_acc']
            }
        except Exception as e:
            print(f'Error: {e}')
            results_summary[f'exp4_{tag}'] = {'status': 'failed', 'error': str(e)}

    print(f'\n{"="*60}')
    print('EXPERIMENT SUMMARY')
    print(f'{"="*60}')
    for exp, result in results_summary.items():
        if result['status'] == 'success':
            print(f'{exp}: Top-1 = {result["top1_acc"]:.2f}%  Top-5 = {result["top5_acc"]:.2f}%')
        elif result['status'] == 'skipped':
            print(f'{exp}: SKIPPED (already done)')
        elif result['status'] == 'failed':
            print(f'{exp}: FAILED - {result["error"]}')
        else:
            print(f'{exp}: {result["status"]}')


if __name__ == '__main__':
    run_all_experiments()

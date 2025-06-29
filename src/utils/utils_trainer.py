import os
from datetime import datetime


def update_metrics(epochs_metrics, running_metrics): # uprość
    for metric_name in running_metrics:
        if metric_name == 'batch_sizes':# or any(sub in metric_name for sub in ('precision', 'recall', 'f1_score')):
            continue
        if 'precision' in metric_name:
            phase = metric_name.split('_')[0]
            tp = sum(running_metrics[f'{phase}_tp']) / sum(running_metrics['batch_sizes'])
            fp = sum(running_metrics[f'{phase}_fp']) / sum(running_metrics['batch_sizes'])
            metric_values = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metric_name = f'{phase}_precision'
        elif 'recall' in metric_name:
            phase = metric_name.split('_')[0]
            tp = sum(running_metrics[f'{phase}_tp']) / sum(running_metrics['batch_sizes'])
            fn = sum(running_metrics[f'{phase}_fn']) / sum(running_metrics['batch_sizes'])
            metric_values = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metric_name = f'{phase}_recall'
        elif 'f1score' in metric_name:
            phase = metric_name.split('_')[0]
            tp = sum(running_metrics[f'{phase}_tp']) / sum(running_metrics['batch_sizes'])
            fp = sum(running_metrics[f'{phase}_fp']) / sum(running_metrics['batch_sizes'])
            fn = sum(running_metrics[f'{phase}_fn']) / sum(running_metrics['batch_sizes'])
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metric_values = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            metric_name = f'{phase}_f1score'
        else:
            metric_values = sum(running_metrics[metric_name]) / sum(running_metrics['batch_sizes'])
        if metric_name not in epochs_metrics:
            epochs_metrics[metric_name] = []
        epochs_metrics[metric_name].append(metric_values)


def adjust_to_log(metrics: dict[str, list[int]], scope: str, window_start: int, round_at: int=5) -> dict[str, int]: # uprość
    logs = {}
    for metric_name in metrics:
        if metric_name in ('batch_sizes', 'epoch') or (len(metrics[metric_name]) < window_start):   # batch_sizes -> batchsizes ?
            continue
        if metric_name.endswith('@'):
            names = metric_name[:-1].split('_')
            group_name = "_".join(names[:-1])
            metric_name_new = names[-1]
            key_new = f'{scope}_{metric_name_new}_subgroup/{group_name}'
            logs[key_new] = (
                sum(metrics[metric_name][-window_start:])
                / (1 if metric_name.endswith('sizes@') 
                   else sum(metrics[group_name + "_batch_sizes@"][-window_start:]))
                if scope == 'running'
                else metrics[metric_name][-1]
            )
        else:
            names = metric_name.split('_')
            phase_new = "".join(names[:-1])
            metric_name_new = names[-1]
            key_new = f'{scope}_{metric_name_new}/{phase_new}'
            logs[key_new] = (
                sum(metrics[metric_name][-window_start:])
                / sum(metrics['batch_sizes'][-window_start:])
                if scope == 'running'
                else metrics[metric_name][-1]
            )
        logs[key_new] = round(logs[key_new], round_at)
    return logs


def create_paths(base_path, exp_name):
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(os.getcwd(), f'{base_path}/{exp_name}/{date}')
    base_save_path = f'{base_path}/checkpoints'
    os.makedirs(base_save_path)
    return base_path, base_save_path

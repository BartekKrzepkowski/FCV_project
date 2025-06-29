import torch

def accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum()
    acc = correct / y_true.size(0)
    return acc.item()

def mean(lst):
    return sum(lst) / len(lst)


def prf(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    class_pred = predicted.bool()
    class_true = y_true.bool()
    tp = (class_pred & class_true).sum().item()
    fp = (class_pred & ~class_true).sum().item()
    fn = (~class_pred & class_true).sum().item()
    return tp, fp, fn
    
def calc_f1(running_metrics, phase):
    tp = sum(running_metrics[f'{phase}_tp']) / sum(running_metrics['batch_sizes'])
    fp = sum(running_metrics[f'{phase}_fp']) / sum(running_metrics['batch_sizes'])
    fn = sum(running_metrics[f'{phase}_fn']) / sum(running_metrics['batch_sizes'])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1
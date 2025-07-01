import logging
import os
import random
from itertools import product

import numpy as np
import torch


def set_seed(seed=0):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch (CPU & CUDA).

    Args:
        seed (int, optional): Seed value to use. Default is 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_training_artefacts(config, epochs_metrics, save_path):
    """
    Saves the training configuration and training metrics to a file.

    Args:
        config (dict): Training configuration.
        epochs_metrics (dict): Metrics collected during training.
        save_path (str): Path to save the artefacts (usually .pth file).
    """
    to_save = {
        'config': config,
        'metrics': epochs_metrics,
    }
    torch.save(to_save, save_path)


def save_checkpoint(model, optimizer, epoch, save_path="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint zapisano w {save_path}")
    
    
def load_checkpoint(model, optimizer, checkpoint_path="checkpoint.pth"):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logging.info(f"Checkpoint załadowany z epoki {epoch}, loss: {loss:.4f}")
    return epoch, loss


def load_model(model, checkpoint_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def load_optim(optimizer, checkpoint_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return optimizer


def yield_hyperparameters(hyperparams):
    """
    Generator yielding all combinations of hyperparameters.

    Args:
        hyperparams (dict): Dictionary where keys are parameter names and values are lists of possible values.

    Yields:
        tuple: A tuple containing one possible combination of hyperparameters.
    """
    keys = hyperparams.keys()
    yield from product(*(hyperparams[k] for k in keys))
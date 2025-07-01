import logging

import numpy as np
import torch

from src.utils.utils_general import save_checkpoint

class EarlyStopping:
    """
    Implements **early stopping** for training when a monitored metric stops improving.

    Attributes:
        patience (int): Number of epochs to wait without improvement before stopping.
        mode (str): 'min' to monitor a metric that should decrease (e.g., loss), 'max' for metrics that should increase (e.g., F1, accuracy).
        delta (float): Minimum change to qualify as an improvement.
        ckpt (str or None): Path to save the best model checkpoint.
        best_score (float): Best score observed so far.
        counter (int): Counter for epochs without improvement.
        stop (bool): Indicates whether training should be stopped.
        enable_early_stopping (bool): Whether to enable early stopping logic.
        best_epoch (int): Epoch number with the best score.

    Args:
        patience (int, optional): Number of epochs to wait without improvement before stopping. Default is 10.
        mode (str, optional): 'min' or 'max' – which direction means improvement. Default is 'min'.
        delta (float, optional): Minimum change to count as an improvement. Default is 0.0.
        checkpoint_path (str or None, optional): Where to save the best model. If None, no model is saved. Default is None.
        enable_early_stopping (bool, optional): Enable or disable early stopping logic. Default is True.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = 'min',
        delta: float = 0.0,
        checkpoint_path: str | None = None,
        enable_early_stopping: bool = True
    ):
        """
        Initializes the EarlyStopping object.

        Args:
            patience (int, optional): Number of epochs to wait without improvement. Default is 10.
            mode (str, optional): 'min' for decreasing metric (e.g., loss), 'max' for increasing metric (e.g., accuracy). Default is 'min'.
            delta (float, optional): Minimum change to qualify as an improvement. Default is 0.0.
            checkpoint_path (str or None, optional): Path to save the best model checkpoint. Default is None.
            enable_early_stopping (bool, optional): Whether to use early stopping logic. Default is True.
        """
        assert mode in {'min', 'max'}, "mode musi być 'min' lub 'max'"
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.ckpt = checkpoint_path

        self.best_score = np.inf if mode == 'min' else -np.inf
        self.counter = 0
        self.stop = False
        self.enable_early_stopping = enable_early_stopping
        self.best_epoch = 0

    def __call__(self, metric: float, model: torch.nn.Module, optim: torch.optim.Optimizer, epoch: int) -> bool:
        """
        Updates the early stopping logic with the current metric value.

        - Saves a checkpoint if the metric improved.
        - Increments the patience counter if no improvement.
        - Returns True if training should be stopped.

        Args:
            metric (float): Current value of the monitored metric (e.g., validation loss or accuracy).
            model (torch.nn.Module): Model to save if the metric improves.
            optim (torch.optim.Optimizer): Optimizer to save if the metric improves.
            epoch (int): Current epoch number.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        score = -metric if self.mode == 'min' else metric

        if score > self.best_score + self.delta:
            logging.info(f"Poprawa metryki: {self.mode} {self.best_score:.4f} -> {score:.4f} na ep. {epoch}")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.ckpt:
                save_checkpoint(
                    model,
                    optim,
                    epoch,
                    save_path=self.ckpt
                )
                # torch.save(model.state_dict(), self.ckpt)
        else:
            logging.info(f"Brak poprawy metryki: {self.mode} {self.best_score:.4f} (aktualna: {score:.4f}) na ep. {epoch}")
            self.counter += 1
            if self.enable_early_stopping and self.counter >= self.patience:
                self.stop = True
        return self.stop

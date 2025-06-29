import logging

import numpy as np
import torch

from src.utils.utils_general import save_checkpoint

class EarlyStopping:
    """
    Wczesne zatrzymanie treningu, gdy metryka przestaje się poprawiać.

    Parameters
    ----------
    patience : int
        Ile kolejnych epok bez poprawy tolerujemy.
    mode : str
        'min'  – monitorujemy metrykę, która powinna maleć (np. loss)  
        'max'  – monitorujemy metrykę, która powinna rosnąć (np. F1, accuracy)
    delta : float
        Minimalna zmiana uznawana za „poprawę”.
    checkpoint_path : str | None
        Gdzie zapisać najlepszy model; jeśli None, model nie jest zapisywany.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = 'min',
        delta: float = 0.0,
        checkpoint_path: str | None = None,
    ):
        assert mode in {'min', 'max'}, "mode musi być 'min' lub 'max'"
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.ckpt = checkpoint_path

        self.best_score = np.inf if mode == 'min' else -np.inf
        self.counter = 0
        self.stop = False

    def __call__(self, metric: float, model: torch.nn.Module, optim: torch.optim.Optimizer, epoch: int) -> bool:
        """
        Aktualizuje stan.  
        Zwraca True, jeśli należy przerwać trening.
        """
        score = -metric if self.mode == 'min' else metric

        if score > self.best_score + self.delta:
            logging.info(f"Poprawa metryki: {self.mode} {self.best_score:.4f} -> {score:.4f} na ep. {epoch}")
            self.best_score = score
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
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

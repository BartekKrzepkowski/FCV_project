import torch

from src.data.get_datasets import get_voices
from src.modules.cnn import FlexibleCNN

DATASET_NAME_MAP = {
    'voices_spectograms': get_voices
    }

MODEL_NAME_MAP = {
    # 'simple_cnn': SimpleCNN,
    'flexible_cnn': FlexibleCNN,
}

OPTIMIZER_NAME_MAP = {
    'sgd': torch.optim.SGD,
    'adamw': torch.optim.AdamW
}

SCHEDULER_NAME_MAP = {
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'multiplicative': torch.optim.lr_scheduler.MultiplicativeLR,
    'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
}

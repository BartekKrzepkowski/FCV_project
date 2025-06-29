import logging
import pickle
import os

import pandas as pd
import torch
import torchvision.transforms as T
import torchaudio.transforms as AT
from torchvision import datasets
from torch.utils.data import random_split, Subset
from sklearn.model_selection import train_test_split

from src.data.datasets import SpectrogramDataset
import random

def freq_shift(max_bins=2):
    def _fn(spec):
        shift = random.randint(-max_bins, max_bins)
        return torch.roll(spec, shifts=shift, dims=-2)
    return _fn

def db_jitter(low=-2.0, high=2.0):
    def _fn(spec):
        noise = torch.empty_like(spec).uniform_(low, high)
        return spec + noise
    return _fn

def cutout_rect(time_size=10, freq_size=10):
    def _fn(spec):
        t0 = random.randint(0, spec.shape[-1] - time_size - 1)
        f0 = random.randint(0, spec.shape[-2] - freq_size - 1)
        spec[..., f0:f0+freq_size, t0:t0+time_size] = 0.0
        return spec
    return _fn

def get_voices(custom_root, df_train_path, df_val_path, df_test_path, use_transform=False):
    df_train = pd.read_csv(df_train_path)
    df_val = pd.read_csv(df_val_path)
    df_test = pd.read_csv(df_test_path)
    
    # global_mean, global_std = compute_global_mean_std_from_paths(
    #     df_train['filename'].apply(lambda x: os.path.join(custom_root, x)).tolist()
    # )
    # logging.info(f"Global mean: {global_mean}, Global std: {global_std}")
    # mean = [global_mean.item()]    # lista (per-kanał)
    # std  = [global_std.item()]     # lista (per-kanał)

    mean = [-3.332040309906006]    # lista (per-kanał)
    std  = [2.904282331466675]     # lista (per-kanał)
    normalize = T.Normalize(mean=mean, std=std)
    
    spectro_augment = T.Compose([
        AT.FrequencyMasking(freq_mask_param=8),
        AT.TimeMasking(time_mask_param=10),
        freq_shift(max_bins=2),
        db_jitter(low=-2.0, high=2.0),
        cutout_rect(time_size=8, freq_size=8),
        normalize
    ])

    spectro_augment = spectro_augment if use_transform else None

    # Wczytanie zbioru CIFAR-10
    train_dataset = SpectrogramDataset(df_train, custom_root=custom_root, transform=spectro_augment)
    val_dataset = SpectrogramDataset(df_val , custom_root=custom_root)
    test_dataset = SpectrogramDataset(df_test, custom_root=custom_root)
    return train_dataset, val_dataset, test_dataset


from tqdm import tqdm
from typing import List

def compute_global_mean_std_from_paths(paths: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0

    for path in tqdm(paths, desc="Analiza spektrogramów"):
        spec = torch.load(path, weights_only=True).float()

        if spec.dim() == 2:
            spec = spec.unsqueeze(0)  # [1, mel, time]

        total_sum   += spec.sum().item()
        total_sumsq += (spec ** 2).sum().item()
        total_count += spec.numel()

    mean = total_sum / total_count
    var  = (total_sumsq / total_count) - mean**2
    std  = var ** 0.5

    return torch.tensor(mean), torch.tensor(std)

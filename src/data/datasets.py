import logging
import pickle
import random
import os
from pathlib import Path
from collections import defaultdict
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision import datasets


class DatasetClassRemapped(Dataset):
    def __init__(self, base_dataset, class_mapping):
        self.base_dataset = base_dataset
        self.class_mapping = class_mapping
        self.transform = getattr(base_dataset, "transform", None)
        self.targets = base_dataset.targets

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.class_mapping[label]

    def __len__(self):
        return len(self.base_dataset)
    
    
class RemappedSubsetDataset(Dataset):
    def __init__(self, base_dataset, indices, class_mapping, transform=None):
        """
        :param base_dataset: oryginalny dataset (np. ImageFolder, OxfordIIITPet)
        :param indices: lista indeksów (podzbiór)
        :param class_mapping: słownik mapujący oryginalne etykiety -> nowe etykiety
        :param transform: opcjonalna transformacja obrazu
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.class_mapping = class_mapping
        self.transform = transform or getattr(base_dataset, 'transform', None)
        self.targets = [class_mapping[label] for label in np.array(base_dataset.targets)[np.array(indices)]]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.base_dataset[real_idx]
        label = self.class_mapping[label]
        if self.transform:
            image = self.transform(image)
        return image, label


class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, df, custom_root, transform=None):
        self.df = df
        self.custom_root = custom_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename =  os.path.join(self.custom_root, self.df.iloc[idx]['filename'])
        label = self.df.iloc[idx]['label']
        spectrogram = torch.load(filename, weights_only=True).float()
        
        # Ensure spectrogram has channel dimension
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)  # Add channel dimension

        if self.transform:
            spectrogram = self.transform(spectrogram)
            
        return spectrogram, label
    

    
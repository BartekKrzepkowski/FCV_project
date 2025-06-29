import numpy as np
import pickle
import random
import logging

from collections import defaultdict
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10

from src.utils.mapping_new import DATASET_NAME_MAP


def prepare_loaders(data_params):
    train_dataset, val_dataset, test_dataset = DATASET_NAME_MAP[data_params['dataset_name']](**data_params['dataset_params'])
    
    loaders = {
        'train': DataLoader(train_dataset, shuffle=True, **data_params['loader_params']),
        'val': DataLoader(val_dataset, shuffle=False, **data_params['loader_params']),
        'test': DataLoader(test_dataset, shuffle=False, **data_params['loader_params'])
    }
    
    return loaders


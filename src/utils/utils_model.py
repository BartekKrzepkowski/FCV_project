from math import sqrt

import torch
from torch import nn

from src.modules.resnets import ResNet18
from src.utils.utils_general import load_model

from src.utils.mapping_new import MODEL_NAME_MAP


def default_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init_range = 1.0 / sqrt(m.out_features)
        nn.init.uniform_(m.weight, -init_range, init_range)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
def prepare_resnet(model_params):
    return ResNet18(**model_params)


def prepare_model(model_params, init=None):
    model = MODEL_NAME_MAP[model_params['model_name']](**model_params['model_params'])
    if model_params['checkpoint_path'] is not None:
        model = load_model(model, model_params['checkpoint_path'])
    else:
        model.apply(default_init)
    return model


def prepare_linear_model(input_dim, n_classes, hidden_dims=[256, 256, 256, 256, 256]):
    """
    Buduje **głęboką sieć neuronową** (wielo-warstwowy model liniowy) z warstwami ukrytymi.
    
    **Kroki:**
    - Pierwsza warstwa mapuje dane wejściowe do przestrzeni pierwszej warstwy ukrytej.
    - Następnie dodajemy kolejne warstwy ukryte z funkcją aktywacji **ReLU**.
    - Ostatnia warstwa mapuje ostatnią warstwę ukrytą na wyjście (liczba klas).
    """
    layers = []
    # Pierwsza warstwa: wejście -> pierwsza warstwa ukryta
    layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
    # layers.append(nn.ReLU())
    
    # Kolejne warstwy ukryte
    for i in range(1, len(hidden_dims)):
        layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        # layers.append(nn.ReLU())
    
    # Ostatnia warstwa: ostatnia warstwa ukryta -> wyjście
    layers.append(torch.nn.Linear(hidden_dims[-1], n_classes))
    model = torch.nn.Sequential(*layers)
    return model

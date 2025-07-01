import torch

from src.utils.mapping_new import OPTIMIZER_NAME_MAP, SCHEDULER_NAME_MAP
from src.utils.utils_general import load_optim

FORBIDDEN_LAYER_TYPES = [torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d]


def get_every_but_forbidden_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of all parameters in the model that are **not** inside a forbidden layer type.

    Args:
        model (torch.nn.Module): The model to search.
        forbidden_layer_types (list): 
            List of layer types (e.g., [nn.Embedding, nn.LayerNorm]) whose parameters should be excluded.

    Returns:
        list: List of parameter names not belonging to forbidden layers.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_every_but_forbidden_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def configure_optimizer(optim_wrapper, model, optim_kwargs):
    """
    Configures an optimizer with **decoupled weight decay** for specific parameters.

    - Applies weight decay only to parameters **not** in forbidden layers and **not** named 'bias'.
    - Biases and parameters in forbidden layers are excluded from weight decay.

    Args:
        optim_wrapper (callable): Optimizer class (e.g., torch.optim.AdamW).
        model (torch.nn.Module): Model whose parameters will be optimized.
        optim_kwargs (dict): Optimizer keyword arguments, must include 'weight_decay'.

    Returns:
        torch.optim.Optimizer: Instantiated optimizer with grouped parameters.
    """
    weight_decay = optim_kwargs['weight_decay']
    del optim_kwargs['weight_decay']

    decay_parameters = get_every_but_forbidden_parameter_names(model, FORBIDDEN_LAYER_TYPES)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for pn, p in model.named_parameters() if pn in decay_parameters and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for pn, p in model.named_parameters() if pn not in decay_parameters and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim_wrapper(optimizer_grouped_parameters, **optim_kwargs)
    return optimizer


# Funkcja przygotowująca optymalizator
def prepare_optim_and_scheduler(model, optim_scheduler_params):
    """
    Prepares the **optimizer and learning rate scheduler**.

    - Instantiates the optimizer and applies decoupled weight decay to selected parameters.
    - Loads optimizer state from checkpoint if provided.
    - Instantiates the learning rate scheduler if specified.

    Args:
        model (torch.nn.Module): The model to optimize.
        optim_scheduler_params (dict): Dictionary with keys:
            - 'optim_name': str, optimizer name (key for OPTIMIZER_NAME_MAP)
            - 'optim_params': dict, optimizer parameters (must include 'weight_decay')
            - 'checkpoint_path': str or None, optional optimizer checkpoint
            - 'scheduler_name': str or None, scheduler name (key for SCHEDULER_NAME_MAP)
            - 'scheduler_params': dict, scheduler parameters

    Returns:
        tuple: (optimizer, lr_scheduler), where:
            optimizer (torch.optim.Optimizer): Configured optimizer.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler or None): Learning rate scheduler, or None if not set.
    """
    optim_wrapper = OPTIMIZER_NAME_MAP[optim_scheduler_params['optim_name']]
    optim = configure_optimizer(optim_wrapper, model, optim_scheduler_params['optim_params'])
    if optim_scheduler_params['checkpoint_path'] is not None:
        optim = load_optim(optim, optim_scheduler_params['checkpoint_path'])
    lr_scheduler = SCHEDULER_NAME_MAP[optim_scheduler_params['scheduler_name']](optim, **optim_scheduler_params['scheduler_params']) if optim_scheduler_params['scheduler_name'] is not None else None
    return optim, lr_scheduler
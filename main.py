import logging
import os

import torch

from src.trainer.trainer import Trainer
from src.utils.utils_data import prepare_loaders
from src.utils.utils_model import prepare_model
from src.utils.utils_optim import prepare_optim_and_scheduler
from src.modules.early_stopping import EarlyStopping


from src.utils.utils_general import set_seed, yield_hyperparameters


device = "cuda" if torch.cuda.is_available() else "cpu"     


def main(config):
     # ════════════════════════ prepare seed ════════════════════════ #


    set_seed(config.trainer_params['seed'])
    logging.info('Random seed prepared.')


    # ════════════════════════ prepare loaders ════════════════════════ #


    loaders = prepare_loaders(config.data_params)
    logging.info('Loaders prepared.')
    

    # ════════════════════════ prepare model ════════════════════════ #


    model = prepare_model(config.model_params).to(device)
    logging.info('Model prepared.')
    
    
    # ════════════════════════ prepare optimizer & scheduler ════════════════════════ #


    optim, lr_scheduler = prepare_optim_and_scheduler(model, config.optim_scheduler_params)
    logging.info('Optimizer and scheduler prepared.')
    
    
    # ════════════════════════ prepare criterion ════════════════════════ #

    weights = torch.tensor([1.0, 3168/384], device=device)
    criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=weights)
    logging.info('Criterion prepared.')


    # ════════════════════════ prepare extra modules ════════════════════════ #
    

    extra_modules = {}

    early_stopping = EarlyStopping(**config.extra_modules_params['early_stopping'])

    extra_modules['early_stopping'] = early_stopping
    logging.info('Extra modules prepared.')


    # ════════════════════════ prepare trainer ════════════════════════ #
    

    params_trainer = {
        'model': model,
        'criterion': criterion,
        'loaders': loaders,
        'optim': optim,
        'lr_scheduler': lr_scheduler,
        'device': device,
        'extra_modules': extra_modules,
    }
    trainer = Trainer(**params_trainer)
    logging.info('Trainer prepared.')


    # ════════════════════════ train model ════════════════════════ #


    trainer.train_model(config)
    logging.info('Training finished.')


if __name__ == "__main__":
    logging.basicConfig(
            format=(
                '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
            ),
            level=logging.INFO,
            handlers=[logging.StreamHandler()],
            force=True,
        )

    # hyperparameters = {
    #     'optim_name': ['sgd'],
    #     'lr': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
    #     'weight_decay': [0.0],
    # }

    # hyperparameters = {
    #     'optim_name': ['adamw'],
    #     'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    #     'weight_decay': [0.0],
    # }

    # hyperparameters = {
    #     'optim_name': ['sgd'],
    #     'lr': [5e-2],
    #     'scheduler_name': [None, 'cosine','reduce_on_plateau'],
    #     'weight_decay': [0.0, 1e-3, 1e-2, 1e-1],
    # }

    # hyperparameters = {
    #     'optim_name': ['adamw'],
    #     'lr': [1e-3],
    #     'scheduler_name': [None, 'cosine', 'reduce_on_plateau'],
    #     'weight_decay': [0.0, 1e-3, 1e-2, 1e-1],
    # }

    hyperparameters = {
        'optim_name': ['adamw'],
        'lr': [1e-3],
        'scheduler_name': ['reduce_on_plateau'],
        'weight_decay': [1e-1],
        'use_bn': [True],
        'dropout_p': [0.00, 0.1, 0.25],
        'skip': [False, True],
    }

    # hyperparameters = {
    #     'optim_name': ['adamw'],
    #     'lr_scheduler': [5e-3, 1e-2, 5e-2, 1e-1],
    #     'weight_decay': [0.0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    # }

    # hyperparameters = {
    #     'optim_name': ['sgd'],
    #     'lr': [1e-1],
    #     'weight_decay': [1e-2],
    # }

    hyperparameters = {
        'optim_name': ['adamw'],
        'lr': [1e-3],
        'scheduler_name': ['reduce_on_plateau'],
        'weight_decay': [1e-1],
        'use_bn': [True],
        'dropout_p': [0.00],
        'skip': [False],
    }

    activation = 'ReLU'
    skip = False
    dropout_p = 0.00
    use_bn = False
    is_bn_pre_act = False

    scheduler_params = {
        None: None,
        'cosine': {'T_max': 200, 'eta_min': 1e-3, 'verbose': True},
        'reduce_on_plateau': {'mode': 'min', 'factor': 0.8, 'patience': 50, 'verbose': True},
    }


    # for (optim_name, lr, weight_decay) in yield_hyperparameters(hyperparameters):
    for (optim_name, lr, scheduler_name, weight_decay, use_bn, dropout_p, skip) in yield_hyperparameters(hyperparameters):

        class Config:
            trainer_params = {
                'device': device,
                'seed': 83,
                'n_epochs': 1000,
                'exp_name': f'vvoices_detection_{optim_name=}_{lr=}_{weight_decay=}_{scheduler_name=}_{activation=}_{dropout_p=}_{use_bn=}_{is_bn_pre_act=}_{skip=}', # nazwa eksperymentu, która będzie użyta do tworzenia folderów i logowania
                'base_path': os.environ['REPORTS_DIR'],
                'load_checkpoint_path': None,    # saving checkpoint of model and optimizer
                'save_checkpoint_modulo': 50,  # how many epochs to save the model
            }
            data_params = {
                'dataset_name' : 'voices_spectograms',
                'dataset_params': {
                    'custom_root': '/net/pr2/projects/plgrid/plggdnnp/datasets/VOiCES_devkit',
                    'df_train_path': 'data/train_spectogram_df.csv',
                    'df_val_path': 'data/val_spectogram_df.csv',
                    'df_test_path': 'data/test_spectogram_df.csv',
                    'use_transform': True, # if True, then use transforms for the base dataset (per side [CIFAR10 at this point])
                },
                'loader_params': {'batch_size': 128, 'pin_memory': True, 'num_workers': 12}
            }
            model_params = {
                'model_name': 'flexible_cnn',
                'model_params': {
                    'num_classes': 2,
                    'input_height': 80,
                    'input_time': 501,
                    'blocks_cfg': [
                        dict(out_ch=16, use_bn=use_bn, dropout_p=dropout_p, skip=False, activation=activation, is_bn_pre_act=is_bn_pre_act),
                        dict(out_ch=32, use_bn=use_bn, dropout_p=dropout_p, skip=False, activation=activation, is_bn_pre_act=is_bn_pre_act),
                        dict(out_ch=32, use_bn=use_bn, dropout_p=dropout_p, skip=skip, activation=activation, is_bn_pre_act=is_bn_pre_act),
                        ],
                },
                'checkpoint_path': None, # path to load model checkpoint
                'init': None,
                'freeze_backbone': False,  # whether to freeze the backbone of the model
            }
            optim_scheduler_params = {
                'optim_name': optim_name,
                'optim_params': {'lr': lr, 'weight_decay': weight_decay},
                'checkpoint_path': None,  # path to load optimizer state
                'scheduler_name': scheduler_name,
                'scheduler_params': scheduler_params[scheduler_name],
            }
            logger_params = {
                'logger_name': 'wandb',
                'entity': os.environ['WANDB_ENTITY'],
                'project_name': os.environ['WANDB_PROJECT'],
                'mode': 'online',   # używając tego określ również czy logować info na dysk
                # 'hyperparameters': h_params_overall,
            }
            extra_modules_params = {
                'early_stopping': {
                    'patience': 200,  # how many epochs without improvement to wait before stopping
                    'mode': 'max',
                    'delta': 1e-4,
                    'checkpoint_path': f"models/best_model_{optim_name}_{lr}_{weight_decay}_{activation}_{dropout_p}_{use_bn}_{is_bn_pre_act}_{skip}.pt",
                    'enable_early_stopping': False,  # whether to enable early stopping
                }
            }
            
        config = Config()
        main(config)

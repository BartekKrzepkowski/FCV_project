import logging
from typing import Dict

import torch

from src.visualization.wandb_logger import WandbLogger
from src.utils.utils_general import save_checkpoint, save_training_artefacts, load_model
from src.utils.utils_metrics import accuracy, prf, calc_f1
from src.utils.utils_trainer import create_paths, update_metrics, adjust_to_log
from src.utils.utils_visualize import matplotlib_scatters_training, log_to_console, show_and_save_grid

class Trainer:
    def __init__(self, model, criterion, loaders, optim, lr_scheduler, extra_modules, device):
        """
        Args:
            model (torch.nn.Module): model to train
            criterions (Dict): dictionary of criterions for training and evaluation
            loaders (Dict): dictionary of data loaders for different phases (train, val, test)
            optim (torch.optim.Optimizer): optimizer for training
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            extra_modules (Dict): dictionary of additional modules or utilities
            device (str): device to use for training ('cpu' or 'cuda')
        """
        self.model = model
        self.criterion = criterion
        self.loaders = loaders
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.logger = None
        self.base_path = None
        self.base_save_path = None
        self.global_step = None

        self.extra_modules = extra_modules


    def at_exp_start(self, config):
        """
        Initialization of experiment.
        Creates fullname, dirs and logger.
        """
        self.base_path, self.base_save_path = create_paths(config.trainer_params['base_path'], config.trainer_params['exp_name'])
        config.logger_params['log_dir'] = f'{self.base_path}/{config.logger_params["logger_name"]}'
        self.logger = WandbLogger(config.logger_params, exp_name=config.trainer_params['exp_name'])
        
        self.logger.log_model(self.model, self.criterion, log=None)

        # for phase, loader in self.loaders.items():    #POPRAW
        #     show_and_save_grid(loader.dataset, save_path=f"{self.base_path}/{phase}_dataset.png")


    def train_model(self, config):
        '''
        Main training loop.
        Args:
            config (Config): configuration object with all parameters
        '''
        logging.info('Training started.')

        self.at_exp_start(config)
        
        # 1) Przygotowanie metryk
        epochs_metrics = {}
        
        for epoch in range(config.trainer_params['n_epochs']):
            epochs_metrics['epoch'] = epoch # czy musze to zapisywać?

            # 1) Zapisz stanu modelu i optymalizatora
            if (epoch > 0) and (config.trainer_params['save_checkpoint_modulo'] != 0) and (epoch % config.trainer_params['save_checkpoint_modulo'] == 0):
                save_checkpoint(
                    self.model,
                    self.optim,
                    epochs_metrics['epoch'],
                    save_path=f"{self.base_save_path}/checkpoint_epoch_{epochs_metrics['epoch']}.pth"
                )
            
            # 2) Faza treningowa
            self.model.train()
            self.run_phase(epochs_metrics, phase='train', config=config)
            
            # 3) Faza ewaluacji
            with torch.no_grad():
                self.model.eval()
                f1_val = self.run_phase(epochs_metrics, phase='val', config=config)

            # 4) check if early stopping is triggered
            if self.extra_modules['early_stopping'](f1_val, self.model, self.optim, epoch):
                logging.info(f"Early stopping triggered at epoch {epoch}.")
                break

            # 5) Logowanie metryk
            self.at_epoch_end(epoch, epochs_metrics)

            


        if self.extra_modules['early_stopping'].ckpt:
            self.model = load_model(self.model, self.extra_modules['early_stopping'].ckpt)

        self.run_phase(epochs_metrics, phase='test', config=config)
        self.at_epoch_end(epoch, epochs_metrics)
        
        # 4) Logowanie metryk do konsoli
        log_to_console(epochs_metrics)
        
        self.at_exp_end(config, epochs_metrics)
    

    def run_phase(self, epochs_metrics, phase, config):
        logging.info(f'Epoch: {epochs_metrics["epoch"]}, Phase: {phase}.')
        
        running_metrics = {
            f"{phase}_{metric_name}": []
            for metric_name in ('losses', 'accs', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1score')
        }
        running_metrics['batch_sizes'] = []

        batches_per_epoch = len(self.loaders[phase])
        logging.info(f'Batches per epoch: {batches_per_epoch}.')
        self.global_step = epochs_metrics['epoch'] * batches_per_epoch  # czy musi tu być self.?
        config.trainer_params['running_window_start'] = batches_per_epoch // 1
        
        for i, data in enumerate(self.loaders[phase]):
            y_pred, y_true = self.infer_from_data(data, device=config.trainer_params['device'])
            running_metrics = self.gather_batch_metrics(phase, running_metrics, y_pred, y_true)   # cos nie działa podczas testowania

            # ════════════════════════ logging (running) ════════════════════════ #

            if (i + 1) % config.trainer_params['running_window_start'] == 0: # lepsza nazwa na log_multi? słowniek multi = {'log'...}?
                # przygotuj metryki do logowania (running)
                running_logs = adjust_to_log(running_metrics, scope='running', window_start=config.trainer_params['running_window_start'])
                self.log(
                    running_logs,
                    phase,
                    scope='running',
                    step=self.global_step
                )

            self.global_step += 1
        
        update_metrics(epochs_metrics, running_metrics)

        return calc_f1(running_metrics, phase) if phase == 'val' else None
        

    def log(self, scope_logs: Dict, phase: str, scope: str, step: int):
        '''
        Send chosen assets to logger and progress bar
        Args:
            assets (Dict):
            phase:
            scope:
            progress_bar:
        '''
        scope_logs[f'steps/{phase}_{scope}'] = step
        self.logger.log_scalars(scope_logs, step)
        # progress_bar.set_postfix(evaluators_log)

        if self.lr_scheduler is not None and phase == 'train' and scope == 'running':
            self.logger.log_scalars({f'lr_scheduler': self.lr_scheduler.get_last_lr()[0]}, step)


    def at_epoch_end(self, epoch, epochs_metrics):
        logging.info(f'Epoch {epochs_metrics["epoch"]} finished.')
        # ════════════════════════ logging (epoch) ════════════════════════ #
        epoch_logs = adjust_to_log(epochs_metrics, scope='epoch', window_start=0)
        self.log(
            epoch_logs,
            phase='test',
            scope='epoch',
            step=epoch
        )
        
        # 5) Logowanie metryk do konsoli
        log_to_console(epochs_metrics)
        
        
    
    def at_exp_end(self, config, epochs_metrics):
        logging.info('Training finished.')
        save_training_artefacts(
            config,
            epochs_metrics,
            save_path=f"{self.base_save_path}/training_artefacts.pth"
        )

        save_checkpoint(
            self.model,
            self.optim,
            epochs_metrics['epoch'],
            save_path=f"{self.base_save_path}/epoch_{epochs_metrics['epoch']}.pth"
        )
        self.logger.close()
            
        # matplotlib_scatters_training(epochs_metrics, save_path=f"{self.base_path}/metrics.pdf")


    def infer_from_data(self, data, device):
        x_true, y_true = data
        x_true, y_true = x_true.to(device), y_true.to(device)
        y_pred = self.model(x_true)
        return y_pred, y_true
    
    
    def gather_batch_metrics(self, phase, running_metrics, y_pred, y_true):
        loss_list = self.criterion(y_pred, y_true)

        
        loss = loss_list.mean()
            
        if phase == 'train':
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)
        
        acc = accuracy(y_pred, y_true)
        prf_tp, prf_fp, prf_fn = prf(y_pred, y_true)
        precision = prf_tp / (prf_tp + prf_fp) if (prf_tp + prf_fp) > 0 else 0.0
        recall = prf_tp / (prf_tp + prf_fn) if (prf_tp + prf_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        batch_size = y_true.shape[0]

        # ════════════════════════ gathering scalars to logging ════════════════════════ #
        
        
        running_metrics[f'{phase}_losses'].append(loss.item() * batch_size)
        running_metrics[f'{phase}_accs'].append(acc * batch_size)
        running_metrics[f'{phase}_tp'].append(prf_tp * batch_size)
        running_metrics[f'{phase}_fp'].append(prf_fp * batch_size)
        running_metrics[f'{phase}_fn'].append(prf_fn * batch_size)
        running_metrics[f'{phase}_precision'].append(precision * batch_size)
        running_metrics[f'{phase}_recall'].append(recall * batch_size)
        running_metrics[f'{phase}_f1score'].append(f1_score * batch_size)
        running_metrics['batch_sizes'].append(batch_size)

        # for cls in [0, 1]: # trzeba rozszerzyć do niearbitrarnych klas
        #     for corrupted in [True, False]:
        #         mask = (y_true == cls) & (is_corrupted == corrupted)
        #         if mask.sum() == 0: continue  # brak przykładów tej grupy w tym batchu

        #         group_name = f"{phase}_cls{cls}{'cor' if corrupted else 'clean'}"
        #         loss_i = loss_list[mask].mean().item()
        #         acc_i = accuracy(y_pred[mask], y_true[mask])

        #         if group_name + "_losses@" not in running_metrics:
        #             running_metrics[group_name + "_losses@"] = []
        #             running_metrics[group_name + "_accs@"] = []
        #             running_metrics[group_name + "_batch_sizes@"] = []

        #         subgroup_size = mask.sum().item()
        #         running_metrics[group_name + "_losses@"].append(loss_i * subgroup_size)
        #         running_metrics[group_name + "_accs@"].append(acc_i * subgroup_size)
        #         running_metrics[group_name + "_batch_sizes@"].append(subgroup_size)

        return running_metrics
import os

import wandb


class WandbLogger:
    def __init__(self, logger_params, exp_name):
        self.project = logger_params['project_name']
        self.writer = wandb
        self.writer.login(key=os.environ['WANDB_API_KEY'])
        if not os.path.isdir(logger_params['log_dir']):
            os.makedirs(logger_params['log_dir'])
        self.writer.init(
            entity=logger_params['entity'],
            project=logger_params['project_name'],
            name=exp_name,
            # config=dict(config),
            # config=OmegaConf.to_container(config, resolve=True),  # czy nie wystarczy dict(config)?
            dir=logger_params['log_dir'],
            mode=logger_params['mode']
        )

    def close(self):
        self.writer.finish()

    def log_model(self, model, criterion, log, log_freq: int=1000, log_graph: bool=True):
        self.writer.watch(model, criterion, log=log, log_freq=log_freq, log_graph=log_graph)
    
    def log_histograms(self, hists):
        self.writer.log(hists)

    def log_scalars(self, evaluators, step):
        self.writer.log(evaluators)
        
    def log_plots(self, plot_images):
        self.writer.log(plot_images)



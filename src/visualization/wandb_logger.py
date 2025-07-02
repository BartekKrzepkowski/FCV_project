import os

import wandb


class WandbLogger:
    """
    Handles logging with **Weights & Biases (wandb)**, including experiment setup, model monitoring, and metric logging.

    Args:
        logger_params (dict): Parameters for the wandb logger, must include:
            - 'project_name': str, WandB project name
            - 'entity': str, WandB entity/team name
            - 'log_dir': str, Directory for logging outputs
            - 'mode': str, Logging mode ('online', 'offline', etc.)
        exp_name (str): Experiment name for tracking in WandB.
    """
    def __init__(self, logger_params, exp_name):
        """
        Initializes the WandbLogger, creates log directory if necessary, and starts the wandb run.

        Args:
            logger_params (dict): Logger configuration parameters.
            exp_name (str): Experiment name to appear in the WandB dashboard.
        """
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
        """
        Finishes the WandB run and closes the logger.
        """
        self.writer.finish()

    def log_model(self, model, criterion, log, log_freq: int=1000, log_graph: bool=True):
        """
        Watches the model and criterion in WandB for logging gradients and optionally the computation graph.

        Args:
            model (torch.nn.Module): Model to be watched.
            criterion (callable): Loss function.
            log (str or None): What to log (e.g., 'gradients', 'parameters').
            log_freq (int, optional): Logging frequency. Default is 1000.
            log_graph (bool, optional): Whether to log the computation graph. Default is True.
        """
        self.writer.watch(model, criterion, log=log, log_freq=log_freq, log_graph=log_graph)
    
    def log_histograms(self, hists):
        """
        Logs histogram data to WandB.

        Args:
            hists (dict): Dictionary of histogram data to log.
        """
        self.writer.log(hists)

    def log_scalars(self, evaluators, step):
        """
        Logs scalar metrics to WandB.

        Args:
            evaluators (dict): Dictionary of scalar values to log.
            step (int): Global step for logging.
        """
        self.writer.log(evaluators)
        
    def log_plots(self, plot_images):
        """
        Logs plot images to WandB.

        Args:
            plot_images (dict): Dictionary or list of images/plots to log.
        """
        self.writer.log(plot_images)



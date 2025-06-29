import logging
import os
from datetime import datetime

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

def show_and_save_grid(dataset, num_images=16, cols=4, save_path="dataset_grid.png"):
    # wybieramy losowe indeksy obrazów
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    images = [dataset[i][0] for i in indices]
    # labels = [dataset[i][1] for i in indices]

    # tworzymy siatkę
    grid = make_grid(images, nrow=cols, padding=2)

    # konwersja do formatu numpy (dla matplotlib)
    npimg = grid.numpy()
    plt.figure(figsize=(cols*3, (num_images//cols)*3))
    plt.axis('off')
    plt.title('Przykładowe obrazy z datasetu')

    # matplotlib oczekuje formatu (wys, szer, kanały)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    # zapisujemy obraz
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Zapisano obraz jako {save_path}")

    # wyświetlamy
    plt.show()
    
    
def matplotlib_scatters_training(epochs_metrics, save_path):
    print(epochs_metrics.keys())

    subgroups = sorted(set(key.split('_')[1] for key in epochs_metrics.keys() if '@' in key and key != 'epoch'))
    phases = sorted(set(key.split('_')[0] for key in epochs_metrics.keys() if '@' not in key and key != 'epoch'))

    n = len(subgroups)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    for i, subgroup in enumerate(subgroups):
        for metric_name, values in epochs_metrics.items():
            if subgroup in metric_name:
                if 'loss' in metric_name:
                    ax_loss = axes[i, 0]
                    ax_loss.plot(
                        range(len(values)),
                        values,
                        label=metric_name.split('_')[0]   
                    )
                elif 'acc' in metric_name:
                    ax_acc = axes[i, 1]
                    ax_acc.plot(
                        range(len(values)),
                        values,
                        label=metric_name.split('_')[0]  
                    )
        ax_loss.set_title(f"{subgroup} - Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True)
        ax_loss.legend(fontsize=7)
        ax_acc.set_title(f"{subgroup} - Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.grid(True)
        ax_acc.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path[:-4] + '_0.pdf', bbox_inches='tight')
    # fig.close()


    n = len(phases)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    for i, phase in enumerate(phases):
        for metric_name, values in epochs_metrics.items():
            if phase == metric_name.split('_')[0]:
                if 'loss' in metric_name:
                    ax_loss = axes[i, 0]
                    ax_loss.plot(
                        range(len(values)),
                        values,
                        label=metric_name.split('_')[1]  
                    )
                elif 'acc' in metric_name:
                    ax_acc = axes[i, 1]
                    ax_acc.plot(
                        range(len(values)),
                        values,
                        label=metric_name.split('_')[1]  
                    )
        ax_loss.set_title(f"{phase} - Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True)
        ax_loss.legend(fontsize=7)
        ax_acc.set_title(f"{phase} - Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.grid(True)
        ax_acc.legend(fontsize=7)
        
    fig.tight_layout()
    fig.savefig(save_path[:-4] + '_1.pdf', bbox_inches='tight')
    # fig.close()
    
    # Tworzenie wykresu metryk: strata i dokładność
    plt.figure(figsize=(10, 4))
    
    # Wykres strat
    plt.subplot(1, 2, 1)
    for metric_name in epochs_metrics:
        if 'loss' in metric_name:
            plt.plot(
                range(len(epochs_metrics[metric_name])),
                epochs_metrics[metric_name],
                label=metric_name
            )
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True)
    plt.legend()
    
    # Wykres dokładności
    plt.subplot(1, 2, 2)
    for metric_name in epochs_metrics:
        if 'acc' in metric_name:
            plt.plot(
                range(len(epochs_metrics[metric_name])),
                epochs_metrics[metric_name],
                label=metric_name
            )
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.grid(True)
    plt.legend()
        
    plt.savefig(
        save_path,
        bbox_inches='tight'
    )
    # ZMNIEJSZ LEGENDĘ
    
    
def log_to_console(epochs_metrics):
    last_vals = {
        key: vals[-1] if len(vals) > 0 else 0.0
        for key, vals in epochs_metrics.items()
        if key != 'epoch'
    }
    parts_non_groups = [
        f"{key}: {last_vals[key]:.4f}"
        for key in sorted(last_vals)
        if '@' not in key
    ]
    parts_groups = [
        f"{key[:-1]}: {last_vals[key]:.4f}"
        for key in sorted(last_vals)
        if '@' in key
    ]
    metrics_str = ", ".join(parts_non_groups)
    logging.info(f"Epoch {epochs_metrics['epoch']}: {metrics_str}.")
    if len(parts_groups) > 0:
        groups_str = ", ".join(parts_groups)
        logging.info(f"Groups: {groups_str}.")
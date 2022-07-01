import os
import torch
import wandb
import fire
import globals
from models_enum import Models
import pos
import ner

class RunParameters():
    def __init__(self, epochs, lr, batch_size, model):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = model

def main(model = 1, dataset = "pos", epochs = 2, lr = 1e-2, batch_size = 64, project = "probing"):
    model_name = Models.get_run_name(model)
    wandb.init(project=project, name=f"{dataset}_{model_name}")
    params = RunParameters(epochs, lr, batch_size, model)
    wandb.log({"epochs": params.epochs, "lossrate": params.lr, "batch_size": params.batch_size, "dataset": dataset, "model": model_name})

    if not os.path.exists(globals.DATASET_DIR):
        os.mkdir(globals.DATASET_DIR)

    if dataset == "pos":
        pos.go(params)
    elif dataset == "ner":
        ner.go(params)
    elif dataset == "all":
        pos.go(params)
        ner.go(params)

if __name__ == '__main__':
    models = """LINEARPROBEBERT = 1
LINEARPROBERANDOM = 2
LINEARBERT = 3
PROBERESETTEDBERT = 4
MULTILAYERPROBEBERT = 5"""
    print(models)
    fire.Fire(main)
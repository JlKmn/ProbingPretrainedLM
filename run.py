import os
import wandb
import fire
from globals import *
import pos
import ner

def main(dataset, epochs = 1, lossrate = 1e-2, batchsize = 64):
    global EPOCHS, lr, batch_size
    EPOCHS = epochs
    lr = lossrate
    batch_size = batchsize
    wandb.log({"epochs": EPOCHS, "lossrate": lr, "batch_size": batch_size, "dataset": dataset})

    if not os.path.exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)

    if dataset == "pos":
        pos.go()
    elif dataset == "ner":
        ner.go()

if __name__ == '__main__':
    wandb.init(project="probing")
    fire.Fire(main)
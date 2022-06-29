import os
import wandb
import fire
import globals
import pos
import ner

def main(dataset = "pos", epochs = 2):
    wandb.log({"epochs": epochs, "lossrate": globals.lr, "batch_size": globals.batch_size, "dataset": dataset})

    if not os.path.exists(globals.DATASET_DIR):
        os.mkdir(globals.DATASET_DIR)

    if dataset == "pos":
        pos.go(epochs)
    elif dataset == "ner":
        ner.go(epochs)

if __name__ == '__main__':
    wandb.init(project="probing")
    fire.Fire(main)
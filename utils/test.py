import wandb
from utils.accuracy import perf

def test(model, test_loader):
    test_loss, test_accuracy = perf(model, test_loader, "test")
    return test_loss, test_accuracy
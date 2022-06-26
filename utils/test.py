import wandb
from utils.accuracy import perf, ner_perf

def test(model, test_loader):
    test_loss, test_accuracy = perf(model, test_loader, "test")
    return test_loss, test_accuracy

def test_ner(model, test_loader):
    test_loss, test_non0_accuracy, test_total_accuracy = ner_perf(model, test_loader, "test")
    return test_loss, test_non0_accuracy, test_total_accuracy

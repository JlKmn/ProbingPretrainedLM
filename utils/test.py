from globals import TEST_BATCHES
from utils.accuracy import perf

def test(model, test_loader):
    eval_loss, eval_accuracy = perf(model, test_loader, TEST_BATCHES)
    return eval_loss, eval_accuracy
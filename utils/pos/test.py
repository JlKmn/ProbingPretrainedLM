from utils.pos.accuracy import perf

def test(model, test_loader):
    test_loss, test_accuracy = perf(model, test_loader, dataset="test")
    return test_loss, test_accuracy
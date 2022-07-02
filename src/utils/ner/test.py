from src.utils.ner.accuracy import perf

def test(model, test_loader):
    test_loss, test_non0_accuracy, test_total_accuracy = perf(model, test_loader, dataset="test")
    return test_loss, test_non0_accuracy, test_total_accuracy
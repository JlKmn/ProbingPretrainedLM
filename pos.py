from datasets import load_dataset
from models.nn.bertprobe import LinearProbeBert
from models.nn.linearbert import LinearBert
from models.nn.multilayerprobe import MultilayerProbeBert
from models.nn.randomprobe import LinearProbeRandom
from models.nn.resettedbert import ProbeResettedBert
from utils.dataset_loader import load
from utils.helper import GetTotalWordCount
from utils.tokenization import tokenize
from utils.dataloader import data_loader
from utils.test import test
from globals import *
from utils.train import fit

def go():
    train_sentences, train_labels = load_dataset(TRAIN_FILE)
    eval_sentences, eval_labels = load(EVAL_FILE)
    test_sentences, test_labels = load(TEST_FILE)
    print("Training dataset sentences", len(train_sentences))
    print("Training dataset total words", GetTotalWordCount(train_sentences))
    print("Eval dataset sentences", len(eval_sentences))
    print("Eval dataset total words", GetTotalWordCount(eval_sentences))
    print("Test dataset sentences", len(test_sentences))
    print("Test dataset total words", GetTotalWordCount(test_sentences))

    label_vocab['<pad>'] = 0

    train_sentences_ids, train_tagging_ids = tokenize(train_sentences, train_labels)
    eval_sentences_ids, eval_tagging_ids = tokenize(eval_sentences, eval_labels)
    test_sentences_ids, test_tagging_ids = tokenize(test_sentences, test_labels)

    train_loader = data_loader(train_sentences_ids, train_tagging_ids)
    eval_loader = data_loader(eval_sentences_ids, eval_tagging_ids)
    test_loader = data_loader(test_sentences_ids, test_tagging_ids)

    bert_probe_model = LinearProbeBert(len(label_vocab))
    linear_model = LinearProbeRandom(len(label_vocab))
    linear_bert_model = LinearBert(len(label_vocab))
    resetted_bert = ProbeResettedBert(len(label_vocab))
    multilayer_probe_bert = MultilayerProbeBert(len(label_vocab))

    print("TRAINING BERT PROBE")
    fit(bert_probe_model, EPOCHS, train_loader, eval_loader)
    print("TRAINING LINEAR")
    fit(linear_model, EPOCHS, train_loader, eval_loader)
    print("TRAINING BERT LINEAR")
    fit(linear_bert_model, EPOCHS, train_loader, eval_loader)
    print("TRAINING RESETTED BERT PROBE")
    fit(resetted_bert, EPOCHS, train_loader, eval_loader)
    print("TRAINING MULTILAYER BERT PROBE")
    fit(multilayer_probe_bert, EPOCHS, train_loader, eval_loader)

    random_loss, random_accuracy = test(linear_model, test_loader)
    print(f"Random Test: Loss {random_loss} Accuracy {random_accuracy}")
    random_loss, random_accuracy = test(bert_probe_model, test_loader)
    print(f"Bert Probe Test: Loss {random_loss} Accuracy {random_accuracy}")
    random_loss, random_accuracy = test(linear_bert_model, test_loader)
    print(f"Bert Linear Test: Loss {random_loss} Accuracy {random_accuracy}")
    random_loss, random_accuracy = test(resetted_bert, test_loader)
    print(f"Resetted Bert Test: Loss {random_loss} Accuracy {random_accuracy}")
    random_loss, random_accuracy = test(multilayer_probe_bert, test_loader)
    print(f"Multilayer Probe Bert Test: Loss {random_loss} Accuracy {random_accuracy}")
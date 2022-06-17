import os

from models.nn.bertprobe import LinearProbeBert
from models.nn.randomprobe import LinearProbeRandom
from utils.dataset_loader import load
from utils.tokenization import tokenize
from utils.dataloader import data_loader
from utils.test import test
from globals import *
from utils.train import fit

import fire

def main(action = "train", epochs = 1, batches = 2, ebatches = 2, tbatches = 10):
    global EPOCHS, BATCHES, EVAL_BATCHES, TEST_BATCHES
    EPOCHS = epochs
    BATCHES = batches 
    EVAL_BATCHES = ebatches 
    TEST_BATCHES = tbatches
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    if not os.path.exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)
    bert_path = f"{OUT_DIR}/probemodel"
    linear_path = f"{OUT_DIR}/linearmodel"

    train_sentences, train_labels = load(TRAIN_FILE)
    eval_sentences, eval_labels = load(EVAL_FILE)
    test_sentences, test_labels = load(TEST_FILE)

    label_vocab['<pad>'] = 0

    train_sentences_ids, train_tagging_ids = tokenize(train_sentences, train_labels)
    eval_sentences_ids, eval_tagging_ids = tokenize(eval_sentences, eval_labels)
    test_sentences_ids, test_tagging_ids = tokenize(test_sentences, test_labels)

    train_loader = data_loader(train_sentences_ids, train_tagging_ids)
    eval_loader = data_loader(eval_sentences_ids, eval_tagging_ids)
    test_loader = data_loader(test_sentences_ids, test_tagging_ids)

    bert_model = None
    linear_model = None
    if os.path.exists(bert_path):
        print("Loading bert model from disk")
        bert_model = torch.load(bert_path)
    else:
        bert_model = LinearProbeBert(len(label_vocab))

    if os.path.exists(linear_path):
        print("Loading linear model from disk")
        linear_model = torch.load(linear_path)
    else:
        linear_model = LinearProbeRandom(len(label_vocab))

    if action == "train":
        print("TRAINING BERT")
        fit(bert_model, EPOCHS, train_loader, eval_loader)
        print("TRAINING LINEAR")
        fit(linear_model, EPOCHS, train_loader, eval_loader)
        print("Saving model")
        torch.save(bert_model, bert_path)
        torch.save(linear_model, linear_path)
    elif action == "test":
        random_loss, random_accuracy = test(linear_model, test_loader)
        print(f"Random Test: Loss {random_loss} Accuracy {random_accuracy}")
        probe_loss, probe_accuracy = test(bert_model, test_loader)
        print(f"Bert Test: Loss {probe_loss} Accuracy {probe_accuracy}")

if __name__ == '__main__':
    fire.Fire(main)
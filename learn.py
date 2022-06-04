from utils.dataset_loader import load
from utils.tokenization import tokenize
from utils.dataloader import data_loader
from globals import *
import torch

from transformers import AutoTokenizer, AutoModel


def main():
    train_sentences, train_labels = load(TRAIN_FILE)
    eval_sentences, eval_labels = load(EVAL_FILE)
    test_sentences, test_labels = load(TEST_FILE)

    label_vocab['<pad>'] = 0

    train_sentences_ids, train_tagging_ids = tokenize(train_sentences, train_labels)
    eval_sentences_ids, eval_tagging_ids = tokenize(eval_sentences, eval_labels)
    test_sentences_ids, test_tagging_ids = tokenize(test_sentences, test_labels)

    train_loader = data_loader(train_sentences_ids, train_tagging_ids)
    eval_loader = data_loader(eval_sentences_ids, eval_sentences_ids)
    test_loader = data_loader(test_sentences_ids, test_tagging_ids)

if __name__ == '__main__':
    main()
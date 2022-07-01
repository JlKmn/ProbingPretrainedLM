import collections
import torch
from transformers import AutoTokenizer

global DATASET_DIR, TRAIN_FILE, EVAL_FILE, TEST_FILE, device, tokenizer, label_vocab, ner_label_length, DBG_PRINT

DATASET_DIR = "datasets"
TRAIN_FILE = "en_ewt-ud-train.conllu"
EVAL_FILE = "en_ewt-ud-dev.conllu"
TEST_FILE = "en_ewt-ud-test.conllu"

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
device = torch.device(input("What gpu device? "))
label_vocab = collections.defaultdict(lambda: len(label_vocab))

# labels = {'<pad>': 9, 'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
ner_label_length = 10

DBG_PRINT = False

def debug_print(str):
    if DBG_PRINT:
        print(str)


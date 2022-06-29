import collections
import torch
from transformers import AutoTokenizer

DATASET_DIR = "datasets"
TRAIN_FILE = "en_ewt-ud-train.conllu"
EVAL_FILE = "en_ewt-ud-dev.conllu"
TEST_FILE = "en_ewt-ud-test.conllu"

device = torch.device('cuda:3')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

label_vocab = collections.defaultdict(lambda: len(label_vocab))

batch_size = 64
EPOCHS = 1
lr = 1e-2

# labels = {'<pad>': 9, 'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
ner_label_length = 10

DBG_PRINT = False

def debug_print(str):
    if DBG_PRINT:
        print(str)
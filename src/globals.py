import collections
import torch
from transformers import AutoTokenizer
import sys


global DATASET_DIR, TRAIN_FILE, EVAL_FILE, TEST_FILE, device, tokenizer, label_vocab, ner_label_length, DBG_PRINT

DATASET_DIR = "datasets"
TRAIN_FILE = "en_ewt-ud-train.conllu"
EVAL_FILE = "en_ewt-ud-dev.conllu"
TEST_FILE = "en_ewt-ud-test.conllu"

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
device = None
if not "pytest" in sys.modules:
    device = torch.device(input("What pytorch device? "))
label_vocab = collections.defaultdict(lambda: len(label_vocab))

# notice that all beginning entities have an odd index
ner_labels = { 9:'<pad>',  0:'O',  1:'B-PER',  2:'I-PER',  3:'B-ORG',  4:'I-ORG',  5:'B-LOC',  6:'I-LOC',  7:'B-MISC',  8:'I-MISC'}
ner_label_length = 10

DBG_PRINT = False

def debug_print(str):
    if DBG_PRINT:
        print(str)


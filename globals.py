import collections
import torch
from transformers import AutoTokenizer

TRAIN_FILE = "en_ewt-ud-train.conllu"
EVAL_FILE = "en_ewt-ud-dev.conllu"
TEST_FILE = "en_ewt-ud-test.conllu"

device = torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

label_vocab = collections.defaultdict(lambda: len(label_vocab))

batch_size = 64
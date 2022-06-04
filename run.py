import os
from models.nn.bertprobe import LinearProbeBert
from models.nn.randomprobe import LinearProbeRandom
from utils.dataset_loader import load
from utils.tokenization import tokenize
from utils.dataloader import data_loader
from globals import *
from utils.train import fit

def main():
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    if not os.path.exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)
    output_path = f"{OUT_DIR}/probemodel"

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
    if os.path.exists(output_path):
        print("Loading model from disk")
        bert_model = torch.load(output_path)
    else:
        bert_model = LinearProbeBert(len(label_vocab))

    random_model = LinearProbeRandom(len(label_vocab))
    fit(bert_model, EPOCHS, train_loader, eval_loader)
    fit(random_model, EPOCHS, train_loader, eval_loader)
    print("Saving model")
    torch.save(bert_model, output_path)

if __name__ == '__main__':
    main()
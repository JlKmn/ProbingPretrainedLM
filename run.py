import os

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

import fire

def main(action = "train", epochs = 1, batches = 2, ebatches = 2, tbatches = 100):
    global EPOCHS, BATCHES, EVAL_BATCHES, TEST_BATCHES
    EPOCHS = epochs
    BATCHES = batches 
    EVAL_BATCHES = ebatches 
    TEST_BATCHES = tbatches
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    if not os.path.exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)
    bert_probe_path = f"{OUT_DIR}/probemodel"
    linear_path = f"{OUT_DIR}/linearmodel"
    bert_linear_path = f"{OUT_DIR}/bertlinearmodel"
    resetted_bert_path = f"{OUT_DIR}/resettedbertmodel"
    multilayer_probe_bert_path = f"{OUT_DIR}/multilayerprobemodel"

    train_sentences, train_labels = load(TRAIN_FILE)
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

    bert_probe_model = None
    if os.path.exists(bert_probe_path):
        print("Loading probe bert model from disk")
        bert_probe_model = torch.load(bert_probe_path)
    else:
        bert_probe_model = LinearProbeBert(len(label_vocab))

    linear_model = None
    if os.path.exists(linear_path):
        print("Loading linear model from disk")
        linear_model = torch.load(linear_path)
    else:
        linear_model = LinearProbeRandom(len(label_vocab))

    linear_bert_model = None
    if os.path.exists(bert_linear_path):
        print("Loading linear bert model from disk")
        linear_bert_model = torch.load(bert_linear_path)
    else:
        linear_bert_model = LinearBert(len(label_vocab))

    resetted_bert = None
    if os.path.exists(resetted_bert_path):
        print("Loading resetted bert model from disk")
        resetted_bert = torch.load(resetted_bert_path)
    else:
        resetted_bert = ProbeResettedBert(len(label_vocab))

    multilayer_probe_bert = None
    if os.path.exists(multilayer_probe_bert_path):
        print("Loading resetted bert model from disk")
        multilayer_probe_bert = torch.load(multilayer_probe_bert_path)
    else:
        multilayer_probe_bert = MultilayerProbeBert(len(label_vocab))

    if action == "train":
        print("TRAINING BERT PROBE")
        fit(bert_probe_model, EPOCHS, train_loader, eval_loader)
        torch.save(bert_probe_model, bert_probe_path)
        print("TRAINING LINEAR")
        fit(linear_model, EPOCHS, train_loader, eval_loader)
        torch.save(linear_model, linear_path)
        print("TRAINING BERT LINEAR")
        fit(linear_bert_model, EPOCHS, train_loader, eval_loader)
        torch.save(linear_bert_model, bert_linear_path)
        print("TRAINING RESETTED BERT PROBE")
        fit(resetted_bert, EPOCHS, train_loader, eval_loader)
        torch.save(resetted_bert, resetted_bert_path)
        print("TRAINING MULTILAYER BERT PROBE")
        fit(multilayer_probe_bert, EPOCHS, train_loader, eval_loader)
        torch.save(multilayer_probe_bert, multilayer_probe_bert_path)
        print("Saving model")
    elif action == "test":
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

# python .\run.py --action test --tbatches 100
# Loading bert model from disk
# Loading linear model from disk
# Random Test: Loss 0.5167459845542908 Accuracy 0.5432551319648093
# Bert Test: Loss 0.10409034043550491 Accuracy 0.8813559322033898
if __name__ == '__main__':
    fire.Fire(main)

# Non-PoS dataset like NER
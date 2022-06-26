from models.nn.bertprobe import LinearProbeBert
from models.nn.linearbert import LinearBert
from models.nn.multilayerprobe import MultilayerProbeBert
from models.nn.randomprobe import LinearProbeRandom
from models.nn.resettedbert import ProbeResettedBert
from utils.dataloader import data_loader
from utils.dataset_loader import load_ner
from globals import EPOCHS, debug_print, ner_label_length
from utils.helper import GetTotalWordCount
from utils.ner_tokenization import tokenize
from utils.train import ner_fit
from utils.test import test_ner

def go():
    train_sentences, train_labels = load_ner("train")
    eval_sentences, eval_labels = load_ner("validation")
    test_sentences, test_labels = load_ner("test")
    print("Training dataset sentences", len(train_sentences))
    print("Training dataset total words", GetTotalWordCount(train_sentences))
    print("Eval dataset sentences", len(eval_sentences))
    print("Eval dataset total words", GetTotalWordCount(eval_sentences))
    print("Test dataset sentences", len(test_sentences))
    print("Test dataset total words", GetTotalWordCount(test_sentences))

    
    train_sentences_ids, train_tagging_ids = tokenize(train_sentences, train_labels)
    eval_sentences_ids, eval_tagging_ids = tokenize(eval_sentences, eval_labels)
    test_sentences_ids, test_tagging_ids = tokenize(test_sentences, test_labels)

    train_loader = data_loader(train_sentences_ids, train_tagging_ids)
    eval_loader = data_loader(eval_sentences_ids, eval_tagging_ids)
    test_loader = data_loader(test_sentences_ids, test_tagging_ids)

    bert_probe_model = LinearProbeBert(ner_label_length)
    linear_model = LinearProbeRandom(ner_label_length)
    linear_bert_model = LinearBert(ner_label_length)
    resetted_bert = ProbeResettedBert(ner_label_length)
    multilayer_probe_bert = MultilayerProbeBert(ner_label_length)

    print("TRAINING BERT PROBE")
    ner_fit(bert_probe_model, EPOCHS, train_loader, eval_loader)
    print("TRAINING LINEAR")
    ner_fit(linear_model, EPOCHS, train_loader, eval_loader)
    print("TRAINING BERT LINEAR")
    ner_fit(linear_bert_model, EPOCHS, train_loader, eval_loader)
    print("TRAINING RESETTED BERT PROBE")
    ner_fit(resetted_bert, EPOCHS, train_loader, eval_loader)
    print("TRAINING MULTILAYER BERT PROBE")
    ner_fit(multilayer_probe_bert, EPOCHS, train_loader, eval_loader)

    loss, non0_accuracy, total_accuracy = test_ner(linear_model, test_loader)
    print(f"Random Test: Loss {loss} Non0-Accuracy {non0_accuracy} Total-Accuracy {total_accuracy}")
    loss, non0_accuracy, total_accuracy = test_ner(bert_probe_model, test_loader)
    print(f"Bert Probe Test: Loss {loss} Non0-Accuracy {non0_accuracy} Total-Accuracy {total_accuracy}")
    loss, non0_accuracy, total_accuracy = test_ner(linear_bert_model, test_loader)
    print(f"Full Bert Test: Loss {loss} Non0-Accuracy {non0_accuracy} Total-Accuracy {total_accuracy}")
    loss, non0_accuracy, total_accuracy = test_ner(resetted_bert, test_loader)
    print(f"Resetted Bert Test: Loss {loss} Non0-Accuracy {non0_accuracy} Total-Accuracy {total_accuracy}")
    loss, non0_accuracy, total_accuracy = test_ner(multilayer_probe_bert, test_loader)
    print(f"Multilayer Bert Probe Test: Loss {loss} Non0-Accuracy {non0_accuracy} Total-Accuracy {total_accuracy}")
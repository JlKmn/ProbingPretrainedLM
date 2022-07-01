import wandb
from models_enum import Models
from models.nn.bertprobe import LinearProbeBert
from models.nn.linearbert import LinearBert
from models.nn.multilayerprobe import MultilayerProbeBert
from models.nn.randomprobe import LinearProbeRandom
from models.nn.resettedbert import ProbeResettedBert
from utils.accuracy import ner_perf
from utils.dataloader import data_loader
from utils.dataset_loader import load_ner
from globals import debug_print, ner_label_length, device
from utils.helper import GetTotalWordCount
from utils.ner_tokenization import tokenize
from utils.train import ner_fit
from utils.test import test_ner

def go(params):
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

    train_loader = data_loader(train_sentences_ids, train_tagging_ids, params.batch_size)
    eval_loader = data_loader(eval_sentences_ids, eval_tagging_ids, params.batch_size)
    test_loader = data_loader(test_sentences_ids, test_tagging_ids, params.batch_size)

    if params.model == Models.LPB.value:
        bert_probe_model = LinearProbeBert(ner_label_length)
        print("TRAINING BERT PROBE")
        ner_fit(bert_probe_model, train_loader, eval_loader, params)
        eval_loss, eval_precision, eval_recall = test_ner(bert_probe_model, test_loader)
        print(f"Bert Probe Test: Loss {eval_loss} Precision {eval_precision} Recall {eval_recall}")
    elif params.model == Models.LPR.value:
        linear_model = LinearProbeRandom(ner_label_length)
        print("TRAINING LINEAR RANDOM")
        ner_fit(linear_model, train_loader, eval_loader, params)
        eval_loss, eval_precision, eval_recall = test_ner(linear_model, test_loader)
        print(f"Random Test: Loss {eval_loss} Precision {eval_precision} Recall {eval_recall}")
    elif params.model == Models.LB.value:
        linear_bert_model = LinearBert(ner_label_length)
        print("TRAINING BERT LINEAR")
        ner_fit(linear_bert_model, train_loader, eval_loader, params)
        eval_loss, eval_precision, eval_recall = test_ner(linear_bert_model, test_loader)
        print(f"Full Bert Test: Loss {eval_loss} Precision {eval_precision} Recall {eval_recall}")
    elif params.model == Models.LPRB.value:
        resetted_bert = ProbeResettedBert(ner_label_length)
        print("TRAINING RESETTED BERT PROBE")
        ner_fit(resetted_bert, train_loader, eval_loader, params)
        eval_loss, eval_precision, eval_recall = test_ner(resetted_bert, test_loader)
        print(f"Resetted Bert Test: Loss {eval_loss} Precision {eval_precision} Recall {eval_recall}")
    elif params.model == Models.MPB.value:
        multilayer_probe_bert = MultilayerProbeBert(ner_label_length)
        print("TRAINING MULTILAYER BERT PROBE")
        ner_fit(multilayer_probe_bert, train_loader, eval_loader, params)
        eval_loss, eval_precision, eval_recall = test_ner(multilayer_probe_bert, test_loader)
        print(f"Multilayer Bert Probe Test: Loss {eval_loss} Precision {eval_precision} Recall {eval_recall}")
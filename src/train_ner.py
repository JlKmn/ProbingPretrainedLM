from src.models.models_enum import Models
from src.models.nn.bertprobe import LinearProbeBert
from src.models.nn.linearbert import LinearBert
from src.models.nn.multilayerprobe import MultilayerProbeBert
from src.models.nn.randomprobe import LinearProbeRandom
from src.models.nn.resettedbert import ProbeResettedBert
from src.utils.shared.dataloader import data_loader
from src.utils.ner.dataset_loader import load
from src.utils.helper import GetTotalWordCount
from src.utils.ner.tokenization import tokenize
from src.utils.ner.train import fit
from src.utils.ner.test import test
from src.globals import debug_print, ner_label_length, device

def go(params):
    train_sentences, train_labels = load("train")
    eval_sentences, eval_labels = load("validation")
    test_sentences, test_labels = load("test")
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

    bert_probe_model = LinearProbeBert(ner_label_length)
    eval_loss, eval_precision, eval_recall = test(bert_probe_model, test_loader)

    if params.model == Models.LPB.value:
        bert_probe_model = LinearProbeBert(ner_label_length)
        print("TRAINING BERT PROBE")
        fit(bert_probe_model, train_loader, eval_loader, params)
        eval_loss, eval_precision, eval_recall = test(bert_probe_model, test_loader)
        print(f"Bert Probe Test: Loss {eval_loss} Precision {eval_precision} Recall {eval_recall}")
    elif params.model == Models.LPR.value:
        linear_model = LinearProbeRandom(ner_label_length)
        print("TRAINING LINEAR RANDOM")
        fit(linear_model, train_loader, eval_loader, params)
        eval_loss, eval_precision, eval_recall = test(linear_model, test_loader)
        print(f"Random Test: Loss {eval_loss} Precision {eval_precision} Recall {eval_recall}")
    elif params.model == Models.LB.value:
        linear_bert_model = LinearBert(ner_label_length)
        print("TRAINING BERT LINEAR")
        fit(linear_bert_model, train_loader, eval_loader, params)
        eval_loss, eval_precision, eval_recall = test(linear_bert_model, test_loader)
        print(f"Full Bert Test: Loss {eval_loss} Precision {eval_precision} Recall {eval_recall}")
    elif params.model == Models.LPRB.value:
        resetted_bert = ProbeResettedBert(ner_label_length)
        print("TRAINING RESETTED BERT PROBE")
        fit(resetted_bert, train_loader, eval_loader, params)
        eval_loss, eval_precision, eval_recall = test(resetted_bert, test_loader)
        print(f"Resetted Bert Test: Loss {eval_loss} Precision {eval_precision} Recall {eval_recall}")
    elif params.model == Models.MPB.value:
        multilayer_probe_bert = MultilayerProbeBert(ner_label_length)
        print("TRAINING MULTILAYER BERT PROBE")
        fit(multilayer_probe_bert, train_loader, eval_loader, params)
        eval_loss, eval_precision, eval_recall = test(multilayer_probe_bert, test_loader)
        print(f"Multilayer Bert Probe Test: Loss {eval_loss} Precision {eval_precision} Recall {eval_recall}")
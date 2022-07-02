from torch import tensor
from src.utils.ner.accuracy import ner_spans, recall, precision

def test_ner_spans():
    expected = {(0,0,5), (1,6,3), (8,9,1), (10,11,1), (13,13,5), (14, 16,7)}
    labels = [ 5, 9, 9, 9, 9, 3, 4, 0, 9, 1, 1, 2, 0, 5, 9, 9, 7]
    tensors = [tensor(label) for label in labels]

    actual = ner_spans(tensors)

    assert len(actual) == len(expected)
    assert actual == expected

def test_ner_spans_no_entities():
    labels = [ 9, 9, 9, 0, 9, 9, 0, 0, 9, 9, 0, 9, 9, 9]
    tensors = [tensor(label) for label in labels]

    actual = ner_spans(tensors)

    assert len(actual) == 0

def test_ner_spans_one_big_entity():
    labels = [ 9, 9, 9, 3, 9, 4, 9, 9, 9, 4, 4, 4, 4, 9]
    expected = {(0, len(labels)-2, 3)}
    tensors = [tensor(label) for label in labels]

    actual = ner_spans(tensors)

    assert len(actual) == len(expected)
    assert actual == expected


def test_recall_precision():
    s_pred = {(0,3,3), (5,8,7)}
    s = {(0,3,3), (5,6,7), (7,7,5), (8,8,3)}
    r = recall(s_pred, s)
    p = precision(s_pred, s)

    assert r == 0.25
    assert p == 0.5
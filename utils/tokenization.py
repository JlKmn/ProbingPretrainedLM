import re
import torch
from transformers import AutoTokenizer
from globals import *

def align_to_bert_tokenization(sentences, labels):
    tokenized_sentences = []
    aligned_labels = []

    for s, l in zip(sentences, labels):
        tokenized_sentence = tokenizer.tokenize(' '.join(s))
        aligned_label = []
        current_word = ''
        i = 0
        for token in tokenized_sentence:
            current_word += re.sub(r'^##', '', token)
            s[i] = s[i].replace('\xad', '')
            
            assert token == '[UNK]' or s[i].startswith(current_word)

            if token == '[UNK]' or s[i] == current_word:
                current_word = ''
                aligned_label.append(l[i])
                i += 1
            else:
                aligned_label.append('<pad>')
        
        assert len(tokenized_sentence) == len(aligned_label)

        tokenized_sentences.append(tokenized_sentence)
        aligned_labels.append(aligned_label)
    
    return tokenized_sentences, aligned_labels


def convert_to_ids(sentences, taggings):
  sentences_ids = []
  taggings_ids = []
  for sentence, tagging in zip(sentences, taggings):
    sentence_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(['[CLS]'] + sentence + ['SEP'])).long()
    tagging_tensor = torch.tensor([0] + [label_vocab[tag] for tag in tagging] + [0]).long()

    sentences_ids.append(sentence_tensor.to(device))
    taggings_ids.append(tagging_tensor.to(device))
  return sentences_ids, taggings_ids


def tokenize(sentences, labels):
    bert_tokenized_sentences, aligned_taggings = align_to_bert_tokenization(sentences, labels)
    print(sentences[42])
    print(labels[42])
    sentences_ids, taggings_ids = convert_to_ids(bert_tokenized_sentences, aligned_taggings)
    print(sentences_ids[42])
    print(taggings_ids[42])
    print('num labels:', len(label_vocab))
    return sentences_ids, taggings_ids